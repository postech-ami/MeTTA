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

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas

# Import data readers / generators
from dataset.dataset_mesh import DatasetDream

# Import topology / geometry trainers
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import DLMesh

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render

RADIUS = 2.5

import wandb
from .utils import EasyDict

import cv2
from skimage.transform import resize
import imageio

from render import regularizer

from PIL import Image
from tqdm import tqdm

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

def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, use_normal=False):
    result_dict = {}
    with torch.no_grad():
        lgt.build_mips()
        if FLAGS.camera_space_light:
            lgt.xfm(target['mv'])

        buffers = geometry.render(glctx, target, lgt, opt_material)

        if use_normal:
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

# def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS):

#     # ==============================================================================================
#     #  Validation loop
#     # ==============================================================================================
#     loss_values = []

#     dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)

#     os.makedirs(out_dir, exist_ok=True)
#     with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
#         fout.write('ID, MSE, PSNR\n')

#         if FLAGS.write_video:
#             os.makedirs(os.path.join(FLAGS.out_dir, "video"), exist_ok=True)
#             all_preds = []
#             all_preds_depth = []

#         print("Running validation")
#         for it, target in enumerate(dataloader_validate):

#             # Mix validation background
#             target = prepare_batch(target, FLAGS.background)

#             result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS)
        
#             # for save video
#             if FLAGS.write_video:
#                 _result_image = result_image.clone().detach().cpu().numpy()
#                 _result_image = resize(_result_image, (256, 256))
#                 _result_image = (_result_image * 255).astype(np.uint8)
#                 all_preds.append(_result_image)

#             # Compute metrics
#             opt = torch.clamp(result_dict['opt'], 0.0, 1.0) # [H, W, 3]

#             pred_rgb = opt.permute(2, 0, 1).contiguous().unsqueeze(0) # [1, 3, H, W]
            
#             loss = 0

#             loss_values.append(loss)

#             line = "%d, %1.8f\n" % (it, loss)
#             fout.write(str(line))

#             for k in result_dict.keys():
#                 np_img = result_dict[k].detach().cpu().numpy()
#                 util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

#         if FLAGS.write_video:
#             all_preds = np.stack(all_preds, axis=0)
#             print("[INFO] write validation video...")
#             # imageio.mimwrite(os.path.join(FLAGS.out_dir, "video", "rgb.mp4"), all_preds, fps=25, quality=8, macro_block_size=1)
#             video_save_name = FLAGS.out_dir.split("/")[-1]
#             imageio.mimwrite(os.path.join(FLAGS.out_dir, "video", f"{video_save_name}.mp4"), all_preds, fps=25, quality=8, macro_block_size=1)

#         avg_loss = np.mean(np.array(loss_values))
        
#         line = "AVERAGES: %1.4f\n" % (avg_loss,)
#         fout.write(str(line))

#     return avg_loss


###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, image_loss_fn, text_z, FLAGS):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry # assert true
        self.optimize_light = optimize_light
        self.image_loss_fn = image_loss_fn
        
        self.text_z = text_z
        
        self.FLAGS = FLAGS

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters())

        # self.prepare_embeddings(FLAGS)

    def forward(self, target, it, args):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])

        buffers, mesh, pred_rgb, all_edges, as_latent, iteration, t_iter = self.geometry.tick(self.glctx, target, self.light, self.material, it, args)
        
        return None


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


def inference_mesh(
    glctx,
    geometry,
    opt_material,
    lgt,
    dataset_validate,
    text_z,
    FLAGS,
    pass_idx=0,
    pass_name="",
    optimize_light=False,
    optimize_geometry=False
    ):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================  

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, image_loss_fn, text_z, FLAGS)

    # load latest model
    load_it = trainer_noddp.load()
    if FLAGS.multi_gpu: 
        # Multi GPU training mode
        from torch.nn.parallel import DistributedDataParallel as DDP

        trainer = DDP(trainer_noddp, device_ids=[FLAGS.local_rank], output_device=FLAGS.local_rank, find_unused_parameters=True)
        trainer.eval()

    else:
        # Single GPU training mode
        trainer = trainer_noddp
        trainer.eval()

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    # dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)
    # dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)

    ###
    loss_values = []


    os.makedirs(FLAGS.out_dir, exist_ok=True)
    with open(os.path.join(FLAGS.out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')

        if FLAGS.write_video:
            os.makedirs(os.path.join(FLAGS.out_dir, f"inference_{FLAGS.infer_idx}"), exist_ok=True)
            os.makedirs(os.path.join(FLAGS.out_dir, f"video_{FLAGS.infer_idx}"), exist_ok=True)
            all_preds = []
            all_preds_depth = []

        print("Running validation")
        for it, target in enumerate(tqdm(dataloader_validate, "[inference]")):

            # Mix validation background
            target = prepare_batch(target, FLAGS.background)

            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS, FLAGS.use_normal)
            
            # for save video
            if FLAGS.write_video:
                _result_image = result_image.clone().detach().cpu().numpy()
                _result_image = resize(_result_image, (256, 256))
                _result_image = (_result_image * 255).astype(np.uint8)
                all_preds.append(_result_image)

            # W, H, _ = _result_image.shape
            pil_img = Image.fromarray(_result_image)
            # pil_img = resize(pil_img, (256, 256))
            pil_img.save(os.path.join(FLAGS.out_dir, f"inference_{FLAGS.infer_idx}", f"novel_{it:06d}.png"))

            # Compute metrics
            opt = torch.clamp(result_dict['opt'], 0.0, 1.0) # [H, W, 3]

            pred_rgb = opt.permute(2, 0, 1).contiguous().unsqueeze(0) # [1, 3, H, W]
            
            loss = 0

            loss_values.append(loss)

            line = "%d, %1.8f\n" % (it, loss)
            fout.write(str(line))

            # for k in result_dict.keys():
            #     np_img = result_dict[k].detach().cpu().numpy()
            #     util.save_image(FLAGS.out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

        if FLAGS.write_video:        
            all_preds = np.stack(all_preds, axis=0)
            print("[INFO] write validation video...")
            # imageio.mimwrite(os.path.join(FLAGS.out_dir, "video", "rgb.mp4"), all_preds, fps=25, quality=8, macro_block_size=1)
            video_save_name = FLAGS.out_dir.split("/")[-1]
            imageio.mimwrite(os.path.join(FLAGS.out_dir, f"video_{FLAGS.infer_idx}", f"{video_save_name}.mp4"), all_preds, fps=25, quality=8, macro_block_size=1)

        avg_loss = np.mean(np.array(loss_values))
        
        line = "AVERAGES: %1.4f\n" % (avg_loss,)
        fout.write(str(line))

    ###
    return None