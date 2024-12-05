# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import glob
import time
import argparse
import json
import itertools
import torch.nn.functional as F

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render

import wandb
from .utils import EasyDict

import cv2
from skimage.transform import resize
import imageio
from PIL import Image

from render import regularizer

import random

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Loss setup
###############################################################################

@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

###############################################################################
# Mix background into a dataset image
###############################################################################

@torch.no_grad()
def prepare_batch(target, bg_type='black'):

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()

    return target

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS):
    eval_mesh, _ = geometry.getMesh(mat)
    
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'])
    
    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : mat['bsdf'],
        'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh

###############################################################################
# Utility functions for material
###############################################################################

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None, bsdf='pbr'):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0) # [9,]
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0) # [9,]
        # input 3d coord [..., 3], output 9-channel texture in mlp_min~mlp_max [..., 9]
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max])
        mat =  material.Material({'kd_ks_normal' : mlp_map_opt})
    else:
        # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
        if FLAGS.random_textures or init_mat is None:
            num_channels = 4 if FLAGS.layers > 1 else 3
            kd_init = torch.rand(size=FLAGS.texture_res + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init , FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # Setup normal map
        if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = bsdf

    return mat

###############################################################################
# Validation & testing
###############################################################################

def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, iteration, use_normal=False):
    result_dict = {}
    with torch.no_grad():
        lgt.build_mips()
        if FLAGS.camera_space_light:
            lgt.xfm(target['mv'])

        buffers = geometry.render(glctx, target, lgt, opt_material)

        if use_normal and (FLAGS.geo_range[0] <= iteration) and (iteration <= FLAGS.geo_range[1]):
            # result_dict['opt'] = util.rgb_to_srgb(buffers['normal'][...,0:3])[0]
            result_dict['opt'] = buffers['normal'][...,0:3][0].contiguous()
        else:
            # result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
            result_dict['opt'] = buffers['shaded'][...,0:3][0].contiguous()
        result_image = result_dict['opt']
        result_mask = buffers['shaded'][..., 3]  # [B, 1, H, W]
        
        # to make image with white background
        # result_image = result_image.permute(2, 0, 1).contiguous()
        result_mask = result_mask.permute(1, 2, 0)
        result_image = result_image * result_mask + (1 - result_mask) * 1  # white bg
        
        if FLAGS.display is not None:
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                elif 'relight' in layer:
                    if not isinstance(layer['relight'], light.EnvironmentLight):
                        layer['relight'] = light.load_env(layer['relight'])
                    img = geometry.render(glctx, target, layer['relight'], opt_material)
                    result_dict['relight'] = util.rgb_to_srgb(img[..., 0:3])[0]
                    result_image = torch.cat([result_image, result_dict['relight']], axis=1)
                elif 'bsdf' in layer:
                    buffers = geometry.render(glctx, target, lgt, opt_material, bsdf=layer['bsdf'])
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])  
                    elif layer['bsdf'] == 'normal':
                        result_dict[layer['bsdf']] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                    else:
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)
   
        return result_image, result_dict

def validate(glctx, geometry, opt_material, lgt, dataset_validate, dataset_inference, out_dir, FLAGS):

    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    loss_values = []

    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)
    dataloader_inference = torch.utils.data.DataLoader(dataset_inference, batch_size=1, collate_fn=dataset_validate.collate)

    inf_it = itertools.cycle(dataloader_inference)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')

        if FLAGS.write_video:
            os.makedirs(os.path.join(FLAGS.out_dir, "video"), exist_ok=True)
            all_preds = []
            all_preds_depth = []

        print("Running validation")
        for it, target in enumerate(dataloader_validate):

            # Mix validation background
            target = prepare_batch(target, FLAGS.background)

            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, it)
            inf_image, inf_dict = validate_itr(glctx, prepare_batch(next(inf_it)), geometry, opt_material, lgt, FLAGS, it)
            
            # for save video
            if FLAGS.write_video:
                _result_image = result_image.clone().detach().cpu().numpy()
                _result_image = resize(_result_image, (256, 256))
                _result_image = (_result_image * 255).astype(np.uint8)
                all_preds.append(_result_image)
            
            _inf_image = inf_image.clone().detach().cpu().numpy()
            _inf_image = resize(_inf_image, (256, 256))
            _inf_image = (_inf_image * 255).astype(np.uint8)
            
            W, H, _ = _inf_image.shape
            # pil_img = np.clip(np.rint(_inf_image * 255.0), 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(_inf_image)
            # pil_img = pil_img.resize((H, W))
            pil_img.save(os.path.join(FLAGS.out_dir, "inference", f"novel_{it:06d}.png"))


            # Compute metrics
            opt = torch.clamp(result_dict['opt'], 0.0, 1.0) # [H, W, 3]

            pred_rgb = opt.permute(2, 0, 1).contiguous().unsqueeze(0) # [1, 3, H, W]
            
            loss = 0

            loss_values.append(loss)

            line = "%d, %1.8f\n" % (it, loss)
            fout.write(str(line))

            # remove dmtet_validate
            # for k in result_dict.keys():
            #     np_img = result_dict[k].detach().cpu().numpy()
            #     util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

        if FLAGS.write_video:
            all_preds = np.stack(all_preds, axis=0)
            print("[INFO] write validation video...")
            # imageio.mimwrite(os.path.join(FLAGS.out_dir, "video", "rgb.mp4"), all_preds, fps=25, quality=8, macro_block_size=1)
            video_save_name = FLAGS.out_dir.split("/")[-1]
            imageio.mimwrite(os.path.join(FLAGS.out_dir, "video", f"{video_save_name}.mp4"), all_preds, fps=25, quality=8, macro_block_size=1)

            # visualze front view
            vis_out_dir = os.path.join(FLAGS.out_dir, "front")
            all_vis = glob.glob(os.path.join(vis_out_dir, "*.png"))
            all_vis.sort()
            all_vis = [Image.open(vis) for vis in all_vis]
            imageio.mimwrite(os.path.join(FLAGS.out_dir, "video", f"{video_save_name}_front-view.mp4"), all_vis, fps=10, quality=8, macro_block_size=1)

            # visualze back view
            vis_out_dir = os.path.join(FLAGS.out_dir, "back")
            all_vis = glob.glob(os.path.join(vis_out_dir, "*.png"))
            all_vis.sort()
            all_vis = [Image.open(vis) for vis in all_vis]
            imageio.mimwrite(os.path.join(FLAGS.out_dir, "video", f"{video_save_name}_back-view.mp4"), all_vis, fps=10, quality=8, macro_block_size=1)


        avg_loss = np.mean(np.array(loss_values))
        
        line = "AVERAGES: %1.4f\n" % (avg_loss,)
        fout.write(str(line))

    return avg_loss


###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, image_loss_fn, guidance_model, text_z, embeddings, FLAGS, target_params):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry # assert true
        self.optimize_light = optimize_light
        self.image_loss_fn = image_loss_fn
        self.guidance_model = guidance_model
        self.text_z = text_z
        self.embeddings = embeddings
        self.FLAGS = FLAGS
        self.target_params = target_params

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters())

        if self.FLAGS.train_location:
            self.loc_params = [self.target_params.azimuth, self.target_params.elevation]

        self.prepare_embeddings(FLAGS)

    def forward(self, target, it, args, target_pixel=None):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])

        if (self.FLAGS.change_light) and (not self.optimize_light):
            # env_path = glob.glob(f"data/irrmaps/rot_hdr/*.hdr")
            # env_path.sort()
            self.light = light.load_env(random.choice(self.FLAGS.envmap_path), scale=self.FLAGS.env_scale, FLAGS=self.FLAGS)
            with torch.no_grad():
                self.light.build_mips()

        buffers, mesh, pred_rgb, pred_ws, all_edges, as_latent, iteration, t_iter = self.geometry.tick(self.glctx, target, self.light, self.material, it, False, args)
        if target_pixel:
            buffers_pixel, _, pred_rgb_pixel, pred_ws_pixel, _, _, _, _ = self.geometry.tick(self.glctx, target_pixel, self.light, self.material, it, True, args)
        
        losses = dict()

        # diffusion SDS loss
        # if self.FLAGS.text:
        if 'SD' in self.guidance_model:
            text_z = self.embeddings['SD']['default']
            if self.FLAGS.directional_text:
                all_pos = []
                all_neg = []
                for emb in text_z[target['direction']]: # list of [2, S, -1]
                    pos, neg = emb.chunk(2) # [1, S, -1]
                    all_pos.append(pos)
                    all_neg.append(neg)
                text_embedding = torch.cat(all_pos + all_neg, dim=0) # [2b, S, -1]
            else:
                text_embedding = text_z

            img_loss = self.guidance_model['SD'].train_step(text_embedding, pred_rgb.half(), as_latent=as_latent, args=args)
            losses['img_loss'] = img_loss

        # if self.FLAGS.image:
        if 'zero123' in self.guidance_model:
            polar = target['polar']
            azimuth = target['azimuth']
            radius = target['radius']
            
            # No need lower codes
            # polar = torch.FloatTensor([torch.rad2deg(p) for p in polar]).unsqueeze(1)
            # azimuth = torch.FloatTensor([torch.rad2deg(a) - 180 for a in azimuth]).unsqueeze(1)

            lambda_guidance = (abs(azimuth) / 180) * self.FLAGS.lambda_guidance

            img_loss = self.guidance_model['zero123'].train_step(self.embeddings['zero123']['default'], pred_rgb, polar, azimuth, radius=radius, as_latent=as_latent, guidance_scale=self.FLAGS.guidance_scale, grad_scale=lambda_guidance, args=args)
            losses['img_loss'] = img_loss

        if (not as_latent) and ('clip' in self.guidance_model):
            azimuth = target['azimuth']

            lambda_guidance = 10 * (1 - abs(azimuth) / 180) * self.FLAGS.lambda_guidance
            clip_loss = self.guidance_model['clip'].train_step(self.embeddings['clip'], pred_rgb, grad_scale=lambda_guidance)
            img_loss += clip_loss
            losses['img_loss'] = img_loss
        # img_loss = torch.tensor(0.0, device = "cuda")

        # below are lots of regularizations...
        reg_loss = torch.tensor(0.0, device = "cuda")

        if self.FLAGS.image:
            ### SDF regularizer
            if (int(self.FLAGS.geo_range[0]) <= iteration) and (iteration <= int(self.FLAGS.geo_range[1])):
                # SDF regularizer
                sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01) * min(1.0, 4.0 * t_iter)
                sdf_loss = sdf_reg_loss(buffers['sdf'], all_edges).mean() * sdf_weight # Dropoff to 0.01
                sdf_loss = sdf_loss * self.FLAGS.lambda_mesh_normal
                reg_loss = reg_loss + sdf_loss

                # directly regularize mesh smoothness in finetuning...
                # if iteration > int(self.FLAGS.iter * 0.2):
                if iteration > int(self.FLAGS.geo_range[0] + (self.FLAGS.geo_range[1] - self.FLAGS.geo_range[0]) * self.FLAGS.geo_schedule):
                    lap_loss = regularizer.laplace_regularizer_const(mesh.v_pos, mesh.t_pos_idx) * self.FLAGS.laplace_scale #* min(1.0, iteration / 500)
                    lap_loss = lap_loss * self.FLAGS.lambda_mesh_laplacian
                    reg_loss = reg_loss + lap_loss
                
                # normal_loss = regularizer.normal_consistency(mesh.v_pos, mesh.t_pos_idx) * self.FLAGS.laplace_scale * min(1.0, iteration / 500)
                # reg_loss = reg_loss + normal_loss
            
            if (int(self.FLAGS.tex_range[0]) <= iteration) and (iteration <= int(self.FLAGS.tex_range[1])):
                # Albedo (k_d) smoothnesss regularizer
                # reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, (iteration - int(self.FLAGS.iter * 0.6)) / 500)

                # # Visibility regularizer
                # reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, (iteration - int(self.FLAGS.iter * 0.6)) / 500)

                # # Light white balance regularizer
                reg_loss += self.light.regularizer() * 0.005
        else:
            if iteration < int(self.FLAGS.iter * 0.6):
                # SDF regularizer
                sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01) * min(1.0, 4.0 * t_iter)
                sdf_loss = sdf_reg_loss(buffers['sdf'], all_edges).mean() * sdf_weight # Dropoff to 0.01
                sdf_loss = sdf_loss * self.FLAGS.lambda_mesh_normal
                reg_loss = reg_loss + sdf_loss

                # directly regularize mesh smoothness in finetuning...
                if iteration > int(self.FLAGS.iter * 0.2):
                    lap_loss = regularizer.laplace_regularizer_const(mesh.v_pos, mesh.t_pos_idx) * self.FLAGS.laplace_scale #* min(1.0, iteration / 500)
                    lap_loss = lap_loss * self.FLAGS.lambda_mesh_laplacian
                    reg_loss = reg_loss + lap_loss
                
                # normal_loss = regularizer.normal_consistency(mesh.v_pos, mesh.t_pos_idx) * self.FLAGS.laplace_scale * min(1.0, iteration / 500)
                # reg_loss = reg_loss + normal_loss
            
            else:
                # Albedo (k_d) smoothnesss regularizer
                # reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, (iteration - int(self.FLAGS.iter * 0.6)) / 500)

                # # Visibility regularizer
                # reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, (iteration - int(self.FLAGS.iter * 0.6)) / 500)

                # # Light white balance regularizer
                reg_loss += self.light.regularizer() * 0.005
        
        if target_pixel and self.FLAGS.image and self.FLAGS.pixel_loss:
            gt_rgb = self.rgb  # [1, 3, 512, 512]
            gt_mask = self.mask  # [1, 512, 512]
            gt_normal = self.normal  # [1, 512, 512, 3]
            gt_depth = self.depth  # [1, 512, 512])
            
            _loss_fn = torch.nn.L1Loss()

            # color loss
            # gt_rgb = gt_rgb * gt_mask[:, None].float() + (1 - gt_mask[:, None].float())  # white bg
            # FIXME: L1 loss
            # rgb_loss = F.mse_loss(pred_rgb_pixel, gt_rgb)
            rgb_loss = _loss_fn(pred_rgb_pixel, gt_rgb)

            # FIXME:mask loss - cross entropy or L1
            # mask_loss = F.mse_loss(pred_ws_pixel[:, 0], gt_mask.float())
            mask_loss = _loss_fn(pred_ws_pixel[:, 0], gt_mask.float())

            # FIXME: set rgb loss higher
            # if self.FLAGS.hard_time and (iteration >= 0) and (iteration <= 300):
            #     losses['rgb_loss'] = rgb_loss * (self.FLAGS.lambda_rgb * 100)
            #     losses['mask_loss'] = mask_loss * (self.FLAGS.lambda_mask * 100)
            # else:
            #     losses['rgb_loss'] = rgb_loss * self.FLAGS.lambda_rgb
            #     losses['mask_loss'] = mask_loss * self.FLAGS.lambda_mask
            
            losses['rgb_loss'] = rgb_loss * self.FLAGS.lambda_rgb
            losses['mask_loss'] = mask_loss * self.FLAGS.lambda_mask      
            

        # FIXME: reg loss weight
        losses['reg_loss'] = reg_loss

        # return img_loss, reg_loss
        return losses
    

    # calculate the text embs.
    @torch.no_grad()
    def prepare_embeddings(self, args):

        # text embeddings (stable-diffusion)
        if args.text is not None:
            if 'SD' in self.guidance_model:
                if args.directional_text:
                    # for d in ['front', 'side', 'back']:
                    #     self.embeddings['SD'][d] = self.guidance_model['SD'].get_text_embeds([f"{args.text}, {d} view"])
                    text_z = []
                    for d in ['front', 'side', 'back', 'side']:
                        # construct dir-encoded text
                        text_z.append(self.guidance_model['SD'].get_text_embeds([f"{args.text}, {d} view"], [''], 1))
                    text_z = torch.stack(text_z, dim=0)
                    
                    self.embeddings['SD']['default'] = text_z

                else:
                    self.embeddings['SD']['default'] = self.guidance_model['SD'].get_text_embeds([args.text], [''], args.batch)
                    # self.embeddings['SD']['uncond'] = self.guidance_model['SD'].get_text_embeds([args.negative])
                
            if 'clip' in self.guidance_model:
                self.embeddings['clip']['text'] = self.guidance_model['clip'].get_text_embeds(args.text, batch=args.batch)
        
        if args.images is not None:

            # h = int(args.known_view_scale * args.h)
            # w = int(args.known_view_scale * args.w)
            # h = int(args.h * args.dmtet_reso_scale)
            # w = int(args.w * args.dmtet_reso_scale)
            h = int(args.dmtet_reso_scale)
            w = int(args.dmtet_reso_scale)

            # load processed image
            for image in args.images:
                assert image.endswith('_rgba.png') # the rest of this code assumes that the _rgba image has been passed.

            rgbas = [cv2.cvtColor(cv2.imread(image, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA) for image in args.images]
            rgba_hw = np.stack([cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in rgbas])
            rgb_hw = rgba_hw[..., :3] * rgba_hw[..., 3:] + (1 - rgba_hw[..., 3:])
            self.rgb = torch.from_numpy(rgb_hw).permute(0,3,1,2).contiguous().cuda()
            self.mask = torch.from_numpy(rgba_hw[..., 3] > 0.5).cuda()
            print(f'[INFO] dataset: load image prompt {args.images} {self.rgb.shape}')

            # load depth
            depth_paths = [image.replace('_rgba.png', '_depth.png') for image in args.images]
            depths = [cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) for depth_path in depth_paths]
            depth = np.stack([cv2.resize(depth, (w, h), interpolation=cv2.INTER_AREA) for depth in depths])
            self.depth = torch.from_numpy(depth.astype(np.float32) / 255).cuda()  # TODO: this should be mapped to FP16
            print(f'[INFO] dataset: load depth prompt {depth_paths} {self.depth.shape}')

            # load normal
            normal_paths = [image.replace('_rgba.png', '_normal.png') for image in args.images]
            normals = [cv2.imread(normal_path, cv2.IMREAD_UNCHANGED) for normal_path in normal_paths]
            normal = np.stack([cv2.resize(normal, (w, h), interpolation=cv2.INTER_AREA) for normal in normals])
            self.normal = torch.from_numpy(normal.astype(np.float32) / 255).cuda()
            print(f'[INFO] dataset: load normal prompt {normal_paths} {self.normal.shape}')

            # encode image_z for zero123
            if 'zero123' in self.guidance_model:
                rgba_256 = np.stack([cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for rgba in rgbas])
                rgbs_256 = rgba_256[..., :3] * rgba_256[..., 3:] + (1 - rgba_256[..., 3:])
                rgb_256 = torch.from_numpy(rgbs_256).permute(0,3,1,2).contiguous().cuda()

                guidance_embeds = self.guidance_model['zero123'].get_img_embeds(rgb_256)
                ###
                self.embeddings['zero123']['default'] = {
                    'zero123_ws' : args.zero123_ws,
                    'c_crossattn' : guidance_embeds[0],  # in list, torch.Size([1, 1, 768])
                    'c_concat' : guidance_embeds[1],  # in list, torch.Size([1, 4, 32, 32])
                    'ref_polars' : args.ref_polars,
                    'ref_azimuths' : args.ref_azimuths,
                    'ref_radii' : args.ref_radii,
                }
            
            if 'clip' in self.guidance_model:
                self.embeddings['clip']['image'] = self.guidance_model['clip'].get_img_embeds(self.rgb, batch=args.batch)

    # save and load checkpoints
    def save(self, it):
        state = {
            'geometry': self.geometry.state_dict(),
            'material': self.material.state_dict(),
            'light': self.light.state_dict() if self.optimize_light else None,
        }
        os.makedirs(os.path.join(self.FLAGS.out_dir, "checkpoints"), exist_ok=True)
        print(f'[INFO] saving checkpoints to {it}.pth')
        torch.save(state, os.path.join(self.FLAGS.out_dir, "checkpoints", f"{it}.pth"))
    
    def load(self, it=None):
        if it is None:
            candidates = [int(os.path.basename(x).split(".")[0]) for x in glob.glob(os.path.join(self.FLAGS.out_dir, "checkpoints", "*.pth"))]
            if len(candidates) == 0: 
                print(f'[INFO] cannot find checkpoints to load')
                return 0
            it = np.max(candidates)
        print(f'[INFO] loading checkpoints from {it}.pth')
        state = torch.load(os.path.join(self.FLAGS.out_dir, "checkpoints", f"{it}.pth"))
        self.material.load_state_dict(state['material'])
        self.geometry.load_state_dict(state['geometry'])
        if state['light'] is not None:
            self.light.load_state_dict(state['light'])

        return it


###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

def eikonal_loss(sdf):
    N = sdf.shape
    sdf_grad = torch.autograd.grad(sdf, sdf, grad_outputs=torch.ones_like(sdf), create_graph=True)[0]
    sdf_grad_norm = torch.norm(sdf_grad, dim=-1)
    eikonal_loss = torch.mean((sdf_grad_norm - 1) ** 2)  # Adjust the target value as needed (1 for unit gradients)
    return eikonal_loss / N

def optimize_mesh(
    glctx,
    geometry,
    opt_material,
    lgt,
    datasets,
    optim,
    guidance, 
    text_z,
    embeddings,
    FLAGS,
    warmup_iter=0,
    log_interval=10,
    pass_idx=0,
    pass_name="",
    optimize_light=True,
    optimize_geometry=True,
    target_params={}
    ):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    # learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    # learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    # learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate

    # def lr_schedule(iter, fraction):
    #     if iter < int(FLAGS.iter * 0.2):
    #         return 1 # 1e-3 geom init
    #     elif iter < int(FLAGS.iter * 0.6):
    #         return 0.1 # 1e-4 geom fine
    #     else:
    #         return 10  # 1e-2 material


    learning_rate_pos = FLAGS.geo_lr
    learning_rate_mat = FLAGS.tex_lr
    if FLAGS.train_location:
        learning_rate_loc = FLAGS.loc_lr

    def geo_lr_schedule(iter, fraction):
        if (int(FLAGS.geo_range[0]) <= iter) and (iter < int(FLAGS.geo_range[1] * FLAGS.geo_schedule)):
            return 1 # 1e-3 geom init
        elif (int(FLAGS.geo_range[1] * FLAGS.geo_schedule) <= iter) and (iter <= int(FLAGS.geo_range[1])):
            return 0.1 # 1e-4 geom fine
        else:
            return 1
    
    def tex_lr_schedule(iter, fraction):
        if (int(FLAGS.tex_range[0]) <= iter) and (iter <= int(FLAGS.tex_range[1])):
            return 1 # 1e-2 material
        else:
            return 1
    
    if FLAGS.train_location:
        def loc_lr_schedule(iter, fraction):
            # FIXME: change adatp location at all training
            # if iter < int(FLAGS.iter * 0.5):
            #     return 1 # 1e-2 material
            # else:
            #     return 0
            return 1
        
        # if iter < warmup_iter:
        #     return iter / warmup_iter 
        # return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, image_loss_fn, guidance, text_z, embeddings, FLAGS, target_params)

    # load latest model
    load_it = trainer_noddp.load()
    if FLAGS.multi_gpu: 
        # Multi GPU training mode
        from torch.nn.parallel import DistributedDataParallel as DDP

        trainer = DDP(trainer_noddp, device_ids=[FLAGS.local_rank], output_device=FLAGS.local_rank, find_unused_parameters=True)
        trainer.train()
    
        if optim == 'adan':
            from optimizer import Adan
            optimizer_mesh = Adan(trainer_noddp.geo_params, lr=learning_rate_pos, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
            optimizer = Adan(trainer_noddp.params, lr=learning_rate_mat, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
            if FLAGS.train_location:
                optimizer_loc = Adan(trainer_noddp.loc_params, lr=learning_rate_loc, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        elif optim == 'adamw':
            optimizer_mesh = torch.optim.AdamW(trainer_noddp.geo_params, lr=learning_rate_pos)
            optimizer = torch.optim.AdamW(trainer_noddp.params, lr=learning_rate_mat)
            if FLAGS.train_location:
                optimizer_loc = torch.optim.AdamW(trainer_noddp.loc_params, lr=learning_rate_loc)

        # optimizer_mesh = torch.optim.AdamW(trainer_noddp.geo_params, lr=learning_rate_pos)
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: geo_lr_schedule(x, 0.9)) 

        # optimizer = torch.optim.AdamW(trainer_noddp.params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: tex_lr_schedule(x, 0.9))
        
        if FLAGS.train_location:
            scheduler_loc = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: loc_lr_schedule(x, 0.9))
    else:
        # Single GPU training mode
        trainer = trainer_noddp
        trainer.train()

        if optim == 'adan':
            from optimizer import Adan
            optimizer_mesh = Adan(trainer_noddp.geo_params, lr=learning_rate_pos, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
            optimizer = Adan(trainer_noddp.params, lr=learning_rate_mat, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
            
            if FLAGS.train_location:
                optimizer_loc = Adan(trainer_noddp.loc_params, lr=learning_rate_loc, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        elif optim == 'adamw':
            optimizer_mesh = torch.optim.AdamW(trainer_noddp.geo_params, lr=learning_rate_pos)
            optimizer = torch.optim.AdamW(trainer_noddp.params, lr=learning_rate_mat)
            
            if FLAGS.train_location:
                optimizer_loc = torch.optim.AdamW(trainer_noddp.loc_params, lr=learning_rate_loc)
        
        # optimizer_mesh = torch.optim.AdamW(trainer_noddp.geo_params, lr=learning_rate_pos)
        scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: geo_lr_schedule(x, 0.9)) 

        # optimizer = torch.optim.AdamW(trainer_noddp.params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: tex_lr_schedule(x, 0.9)) 
        
        if FLAGS.train_location:
            scheduler_loc = torch.optim.lr_scheduler.LambdaLR(optimizer_loc, lr_lambda=lambda x: loc_lr_schedule(x, 0.9)) 

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []

    if FLAGS.pixel_loss:
        rgb_loss_vec = []
        mask_loss_vec = []

    dataset_train = datasets['train']
    dataset_validate = datasets['validate']
    # NOTE: for memory limitation, at inference stage, we don't need this part for visualize.. maybe?!
    dataset_front = datasets['front']
    dataset_back = datasets['back']
    dataset_pixel = datasets['pixel']

    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)
    dataloader_front = torch.utils.data.DataLoader(dataset_front, batch_size=1, collate_fn=dataset_train.collate)
    dataloader_back = torch.utils.data.DataLoader(dataset_back, batch_size=1, collate_fn=dataset_train.collate)
    dataloader_pixel = torch.utils.data.DataLoader(dataset_pixel, batch_size=1, collate_fn=dataset_train.collate)

    v_it = itertools.cycle(dataloader_validate)
    front_it = itertools.cycle(dataloader_front)
    back_it = itertools.cycle(dataloader_back)
    pixel_it = itertools.cycle(dataloader_pixel)

    def save_validate_imgs(my_it, iters, direction_name):
        result_image, result_dict = validate_itr(glctx, prepare_batch(next(my_it), FLAGS.background), geometry, opt_material, lgt, FLAGS, it, use_normal=FLAGS.geo_normal)
        np_result_image = result_image.detach().cpu().numpy()
        if display_image:
            util.display_image(np_result_image, title='%d / %d' % (iters, FLAGS.iter))
        if save_image:
            img_cnt = iters // FLAGS.save_interval
            # util.save_image(FLAGS.out_dir + '/' + ('img_%s_%06d.png' % (pass_name, img_cnt)), np_result_image)
            util.save_image(f"{FLAGS.out_dir}/{direction_name}/img_{iters:06d}.png", np_result_image)

            W, H, _ = np_result_image.shape
            pil_img = np.clip(np.rint(np_result_image * 255.0), 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(pil_img)
            pil_img = pil_img.resize((H // 2, W // 2))
            pil_img.save(os.path.join(FLAGS.out_dir, direction_name, f"img_{iters:06d}.png"))

            # wandb
            if direction_name == "front":
                wandb_logs.update({'Inference/img': wandb.Image(pil_img, caption=f"iters: {iters:06d}"), 'step': iters})


    for it, target in enumerate(dataloader_train):
        # DEBUG
        # if FLAGS.train_location:
        #     print("[target]===")
        #     print(f"azimuth: {target_params['azimuth']}")
        #     print(f"elevation: {target_params['elevation']}")        

        # wandb logging
        if FLAGS.local_rank == 0:
            wandb_logs = EasyDict()

        # load checkpoints
        it += load_it
        if it > FLAGS.iter:
            break

        # Mix randomized background into dataset image
        target = prepare_batch(target, 'random')

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        if FLAGS.local_rank == 0:
            display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
            save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
            if display_image or save_image:
                # result_image, result_dict = validate_itr(glctx, prepare_batch(next(front_it), FLAGS.background), geometry, opt_material, lgt, FLAGS, it, use_normal=FLAGS.geo_normal)
                # np_result_image = result_image.detach().cpu().numpy()
                # if display_image:
                #     util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
                # if save_image:
                #     img_cnt = it // FLAGS.save_interval
                #     # util.save_image(FLAGS.out_dir + '/' + ('img_%s_%06d.png' % (pass_name, img_cnt)), np_result_image)
                #     util.save_image(f"{FLAGS.out_dir}/inference/img_{it:06d}.png", np_result_image)

                #     W, H, _ = np_result_image.shape
                #     pil_img = np.clip(np.rint(np_result_image * 255.0), 0, 255).astype(np.uint8)
                #     pil_img = Image.fromarray(pil_img)
                #     pil_img = pil_img.resize((H // 2, W // 2))
                #     pil_img.save(os.path.join(FLAGS.out_dir, "inference", f"img_{it:06d}.png"))

                #     # wandb
                #     wandb_logs.update({'Inference/img': wandb.Image(pil_img, caption=f"iters: {it:06d}"), 'step': it})
                # save_validate_imgs(v_it, it, "inference")
                save_validate_imgs(front_it, it, "front")
                save_validate_imgs(back_it, it, "back")
                # save_validate_imgs(pixel_it, it, "pixel")

        iter_start_time = time.time()

        # ==============================================================================================
        #  Zero gradients
        # ==============================================================================================
        if (int(FLAGS.geo_range[0]) <= it) and (it <= int(FLAGS.geo_range[1])):
            optimizer_mesh.zero_grad()
        if (int(FLAGS.tex_range[0]) <= it) and (it <= int(FLAGS.tex_range[1])):
            optimizer.zero_grad()
        
        if FLAGS.train_location:
            optimizer_loc.zero_grad()
        # ==============================================================================================
        #  Training
        # ==============================================================================================
        args = EasyDict()
        # args.sds_interval = FLAGS.sds_interval
        # args.out_dir = FLAGS.out_dir
        # args.local_rank = FLAGS.local_rank
        # args.batch = FLAGS.batch
        # args.text = FLAGS.text
        # args.image = FLAGS.image
        args.iteration = it
        if FLAGS.local_rank == 0:
            args.wandb_logs = wandb_logs
        args.update(vars(FLAGS))
        
        # img_loss, reg_loss = trainer(target, prepare_batch(next(pixel_it)), it, args)
        losses = trainer(target, it, args, prepare_batch(next(pixel_it)))
        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        img_loss = losses['img_loss']
        reg_loss = losses['reg_loss']

        total_loss = img_loss + reg_loss

        if FLAGS.pixel_loss:
            rgb_loss = losses['rgb_loss'] * 10  # NOTE: pixel loss weight
            mask_loss = losses['mask_loss'] * 10  # NOTE: pixel loss weight
            total_loss = total_loss + rgb_loss + mask_loss


        img_loss_vec.append(img_loss.item())
        reg_loss_vec.append(reg_loss.item())

        if FLAGS.pixel_loss:
            rgb_loss_vec.append(rgb_loss.item())
            mask_loss_vec.append(mask_loss.item())

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================
        total_loss.backward()

        # if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:  # FIXME: disabled same as fantasia3d
        #     lgt.base.grad *= 64

        # if 'kd_ks_normal' in opt_material:  #FIXME: kd ks weight
        #     opt_material['kd_ks_normal'].encoder.params.grad /= 8.0

        # if 'kd_ks_normal' in opt_material:
        #     opt_material['kd_ks_normal'].encoder.embeddings.grad /= 8.0
        #     opt_material['kd_ks_normal'].encoder.params.grad /= 8.0

        if optim == 'adamw':
            torch.nn.utils.clip_grad_norm_(trainer_noddp.geo_params, FLAGS.grad_clip)
            torch.nn.utils.clip_grad_norm_(trainer_noddp.params, FLAGS.grad_clip)
            if FLAGS.train_location:
                torch.nn.utils.clip_grad_norm_(trainer_noddp.loc_params, FLAGS.grad_clip)

        if (int(FLAGS.geo_range[0]) <= it) and (it <= int(FLAGS.geo_range[1])):
            optimizer_mesh.step()
            scheduler_mesh.step()
        if (int(FLAGS.tex_range[0]) <= it) and (it <= int(FLAGS.tex_range[1])):
            optimizer.step()
            scheduler.step()
        
        if FLAGS.train_location:
            optimizer_loc.step()
            scheduler_loc.step()
        
        # optimizer_mesh.step()
        # scheduler_mesh.step()
        # optimizer.step()
        # scheduler.step()
        
        # ==============================================================================================
        #  Clamp trainables to reasonable range
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.0)

        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Logging
        # ==============================================================================================

        if FLAGS.local_rank == 0:
            wandb_logs.update({'Loss/img': img_loss, 'step': it})
            wandb_logs.update({'Loss/reg': reg_loss, 'step': it})
            wandb_logs.update({'Loss/total': total_loss, 'step': it})

            if FLAGS.pixel_loss:
                wandb_logs.update({'Loss/rgb:': rgb_loss, 'step': it})
                wandb_logs.update({'Loss/mask': mask_loss, 'step': it})
            
            # if FLAGS.train_location:  # FIXME: disabled temporally
            wandb_logs.update({'Train/azimuth': target_params['azimuth'].item(),' step': it})
            wandb_logs.update({'Train/elevation': target_params['elevation'].item(),' step': it})
            
            wandb.log(wandb_logs)

        if it % log_interval == 0 and FLAGS.local_rank == 0:
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
            
            remaining_time = (FLAGS.iter-it)*iter_dur_avg

            if FLAGS.pixel_loss:
                rgb_loss_avg = np.mean(np.asarray(rgb_loss_vec[-log_interval:]))
                mask_loss_avg = np.mean(np.asarray(mask_loss_vec[-log_interval:]))
            
                print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, rgb_loss=%.6f, mask_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                (it, img_loss_avg, reg_loss_avg, rgb_loss_avg, mask_loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))
            
            else:
                print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                    (it, img_loss_avg, reg_loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))
        
        if it > 0 and it % 500 == 0 and FLAGS.local_rank == 0:
            # save model
            trainer_noddp.save(it)
    

    return geometry, opt_material