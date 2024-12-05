# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
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

import wandb

from utils.train import *
from utils.train_cam import optimize_cam
from utils.transform_cam import *
from utils.utils import seed_everything, EasyDict

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display_interval', type=int, default=0)
    parser.add_argument('-si', '--save_interval', type=int, default=100)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    # parser.add_argument('-lr', '--learning_rate', type=float, nargs="*", default=0.001)  # different with config file and CLI... / config: float, CLI: list
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-dt', '--directional-text', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='white', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default="")
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    
    # TODO: newly added args
    parser.add_argument('--envmap', type=str, default=None)
    parser.add_argument('--learn_light', type=bool, default=True)
    parser.add_argument('--wandb_user', type=str, default='ug-kim')
    parser.add_argument('--sds_interval', type=int, default=0)

    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1', 'xl'], help="stable diffusion version")

    parser.add_argument('--image', default=None, help="image prompt")
    parser.add_argument('--known_view_interval', type=int, default=2, help="train default view with RGB loss every & iters, only valid if --image is not None.")
    parser.add_argument('--guidance_scale', type=float, default=100, help="diffusion model classifier-free guidance scale")

    parser.add_argument('--progressive_view', action='store_true', help="progressively expand view sampling range from default to full")
    parser.add_argument('--progressive_level', action='store_true', help="progressively increase gridencoder's max_level")

    # parser.add_argument('--optim_both', action='store_true', default=False)
    parser.add_argument('--desc', type=str, default=None)
    parser.add_argument('--write_video', action='store_true', default=False)

    parser.add_argument('--guidance', type=str, nargs='*', default=['SD'], help='guidance model')
    parser.add_argument('--port', type=str, default="23456")
    parser.add_argument('--optim', type=str, default='adamw', choices=['adan', 'adam', 'adamw'], help="optimizer")

    # config
    parser.add_argument('--out_dir', type=str, default='test')
    parser.add_argument('--text', type=str, default=None)

    parser.add_argument('--seed', type=int, default=5)

    parser.add_argument('--negative', default='', type=str, help="negative text prompt")

    # parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    # parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--dmtet_reso_scale', type=float, default=512, help="multiply --h/w by this for dmtet finetuning")
    
    parser.add_argument('--lambda_guidance', type=float, default=1, help="loss scale for SDS")
    parser.add_argument('--lambda_rgb', type=float, default=1, help="loss scale for RGB")
    parser.add_argument('--lambda_mask', type=float, default=1, help="loss scale for mask (alpha)")
    parser.add_argument('--lambda_normal', type=float, default=0, help="loss scale for normal map")
    parser.add_argument('--lambda_depth', type=float, default=10, help="loss scale for relative depth")
    parser.add_argument('--lambda_2d_normal_smooth', type=float, default=0, help="loss scale for 2D normal image smoothness")
    parser.add_argument('--lambda_3d_normal_smooth', type=float, default=0, help="loss scale for 3D normal image smoothness")

    parser.add_argument('--lambda_mesh_normal', type=float, default=1.0, help="loss scale for mesh normal smoothness")
    parser.add_argument('--lambda_mesh_laplacian', type=float, default=1.0, help="loss scale for mesh laplacian")


    parser.add_argument('--lgt_bias', type=float, default=0.5, help="create_trainable_env_rnd bias")

    parser.add_argument('--lgt_interval', type=int, default=0)
    parser.add_argument('--mat_interval', type=int, default=0)
    parser.add_argument('--mat_stat_log', action='store_true')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--img_debug', action='store_true')
    parser.add_argument('--env_scale', type=float, default=1.0)

    # DMTet mesh initialization
    parser.add_argument('--init_mesh', type=str, default="ellipsoid")
    parser.add_argument('--sdf_init_shape_scale', type=float, nargs="*", default=[1., 1., 1.])
    parser.add_argument("--sdf_init_shape_rotate_x", type= int, nargs=1, default=0 , help="rotation of the initial shape on the x-axis")
    parser.add_argument("--translation_y", type= float, nargs=1, default= 0 , help="translation of the initial shape on the y-axis")

    # zero123
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="stable diffusion time steps range")
    parser.add_argument('--zero123_config', type=str, default='./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml', help="config file for zero123")
    parser.add_argument('--zero123_ckpt', type=str, default='pretrained/zero123/zero123-xl.ckpt', help="ckpt for zero123")
    parser.add_argument('--zero123_grad_scale', type=str, default='angle', help="whether to scale the gradients based on 'angle' or 'None'")

    parser.add_argument('--default_radius', type=float, default=3.2, help="radius for the default view")
    parser.add_argument('--default_polar', type=float, default=90, help="polar for the default view")
    parser.add_argument('--default_azimuth', type=float, default=0, help="azimuth for the default view")
    parser.add_argument('--default_fovy', type=float, default=20, help="fovy for the default view")

    # Hyper-parameters
    parser.add_argument('--geo_lr', type=float, default=0.001)
    parser.add_argument('--tex_lr', type=float, default=0.01)
    parser.add_argument('--loc_lr', type=float, default=0.0001)
    parser.add_argument('--geo_range', type=float, nargs='*', default=[0, 0.6], help="geometry optimization iteration range")
    parser.add_argument('--tex_range', type=float, nargs='*', default=[0.6, 1], help="texture optimization iteration range")
    parser.add_argument('--geo_normal', action='store_true')
    parser.add_argument('--geo_schedule', type=float, default=0.4)
    parser.add_argument('--pixel_loss', action='store_true')
    parser.add_argument('--pre_iter', type=int, default=5000)
    parser.add_argument('--optim_radius', action='store_true')
    parser.add_argument('--optim_location', action='store_true')
    parser.add_argument('--optim_azimuth', action='store_true')
    parser.add_argument('--optim_elevation', action='store_true')
    parser.add_argument('--train_location', action='store_true')
    parser.add_argument('--train_azimuth', action='store_true')
    parser.add_argument('--train_elevation', action='store_true')
    parser.add_argument('--radius', type=float, default=2.5)
    parser.add_argument('--sdf_regularizer', type=float, default=0.2)

    # debugging
    parser.add_argument('--hard_time', action='store_true')  # it give hard timestep at first 300 iters
    parser.add_argument('--same_time', action='store_true')  # it samples same timestep of an batch at one iter
    parser.add_argument('--init_mesh_thicker', type=float, default=0.)
    parser.add_argument('--pix3d_id', type=str, default="0169")
    parser.add_argument('--pix3d_class', type=str, default="chair")
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--data_type', type=str, default="pix3d")
    parser.add_argument('--root_path', type=str, default="./")

    parser.add_argument('--change_light', type=bool, default=False)

    # ablation studies
    parser.add_argument('--abla_ellipsoid', action='store_true')
    parser.add_argument('--abla_viewpoint', action='store_true')
    parser.add_argument('--ellipsoid_scale', type=float, default=0.4)

    parser.add_argument('--azimuth_noise', type=float, default=0.0)  # degree
    parser.add_argument('--elevation_noise', type=float, default=0.0)  # degree
    
    FLAGS = parser.parse_args()
    
    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 128                      # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    # FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    # FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    # FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "absolute"               # Mesh Laplacian ["absolute", "relative", "large_steps"]
    FLAGS.laplace_scale       = 10000                  # Weight for sdf regularizer. Default is relative with large weight
    # FLAGS.normal_scale        = 0.02                  # Weight for sdf regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    # FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    # FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    # FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    # FLAGS.ks_max              = [ 1.0,  1.0,  1.0]

    FLAGS.kd_min              = [ 0.03,  0.03,  0.03] # Limits for kd
    FLAGS.kd_max              = [ 0.97,  0.97,  0.97]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 0,  0.9,  0.9]

    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    # FLAGS.learn_light         = True

    FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    
    FLAGS.geo_range = [int(g * FLAGS.iter) for g in FLAGS.geo_range]
    FLAGS.tex_range = [int(t * FLAGS.iter) for t in FLAGS.tex_range]

    if FLAGS.multi_gpu:

        # adjust total iters
        FLAGS.iter = int(FLAGS.iter / int(os.environ["WORLD_SIZE"]))

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            # os.environ["MASTER_PORT"] = '23456'
            os.environ["MASTER_PORT"] = FLAGS.port

        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]
    
    # if FLAGS.display_res is None:
    #     FLAGS.display_res = FLAGS.train_res

    # make workspace numbering
    # if (FLAGS.local_rank == 0) and (not FLAGS.debug):
    #     prev_run_dirs = glob.glob("work_dirs/*")
    #     prev_run_ids = [int(p.split("/")[-1].split("_")[0]) for p in prev_run_dirs]
    #     cur_run_id = max(prev_run_ids, default=-1) + 1
    #     FLAGS.out_dir = f"{cur_run_id:04d}_{FLAGS.out_dir}"
    #     if FLAGS.desc != None:
    #         FLAGS.out_dir = f"{FLAGS.out_dir}_{FLAGS.desc}"

    # for wandb logging
    FLAGS.out_name = FLAGS.out_dir

    if FLAGS.out_dir is None:
        # FLAGS.out_dir = f'{FLAGS.root_path}/sitto/work_dirs/cube_%d' % (FLAGS.train_res)
        FLAGS.out_dir = os.path.join(FLAGS.root_path, "work_dirs", f'cube_{FLAGS.train_res}')
    else:
        # FLAGS.out_dir = f'{FLAGS.root_path}/work_dirs/' + FLAGS.out_dir
        FLAGS.out_dir = os.path.join(FLAGS.root_path, "work_dirs", FLAGS.out_dir)

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

        os.makedirs(FLAGS.out_dir, exist_ok=True)
        # additional logging dirs
        os.makedirs(os.path.join(FLAGS.out_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(FLAGS.out_dir, "mask"), exist_ok=True)
        os.makedirs(os.path.join(FLAGS.out_dir, "inference"), exist_ok=True)
        os.makedirs(os.path.join(FLAGS.out_dir, "front"), exist_ok=True)
        os.makedirs(os.path.join(FLAGS.out_dir, "back"), exist_ok=True)
        os.makedirs(os.path.join(FLAGS.out_dir, "pixel"), exist_ok=True)
        
        if FLAGS.sds_interval:
            os.makedirs(os.path.join(FLAGS.out_dir, "diffusion"), exist_ok=True)
            os.makedirs(os.path.join(FLAGS.out_dir, "diffusion/latent"), exist_ok=True)
            os.makedirs(os.path.join(FLAGS.out_dir, "diffusion/grad"), exist_ok=True)
            os.makedirs(os.path.join(FLAGS.out_dir, "diffusion/denoised_result"), exist_ok=True)

        # if FLAGS.write_video:
        #     os.makedirs(os.path.join(FLAGS.out_dir, "video"), exist_ok=True)
        
        if FLAGS.lgt_interval:
            os.makedirs(os.path.join(FLAGS.out_dir, "lgt", "diff"), exist_ok=True)
            os.makedirs(os.path.join(FLAGS.out_dir, "lgt", "shaded"), exist_ok=True)
            os.makedirs(os.path.join(FLAGS.out_dir, "lgt", "return"), exist_ok=True)
        
        if FLAGS.mat_interval:
            os.makedirs(os.path.join(FLAGS.out_dir, "mat"), exist_ok=True)


    glctx = dr.RasterizeGLContext()

    # ==============================================================================================
    #  Create Diffusion / CLIP
    # ==============================================================================================

    seed_everything(FLAGS.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if FLAGS.init_mesh_type == "rcnn":
    #     pixel_azimuth, pixel_elevation = meshrcnn_radians("dataset/pix3d.json")
    src_path = os.path.join(FLAGS.root_path, "src", FLAGS.data_type)
    if FLAGS.azimuth_noise and FLAGS.elevation_noise:
        pixel_azimuth, pixel_elevation, mesh_path, image_path = get_radians_noise(src_path, FLAGS.pix3d_class, FLAGS.pix3d_id, np.deg2rad(FLAGS.azimuth_noise), "both")
    elif FLAGS.azimuth_noise:
        pixel_azimuth, pixel_elevation, mesh_path, image_path = get_radians_noise(src_path, FLAGS.pix3d_class, FLAGS.pix3d_id, np.deg2rad(FLAGS.azimuth_noise), "azimuth")
        # FLAGS.train_location = True
    elif FLAGS.elevation_noise:
        pixel_azimuth, pixel_elevation, mesh_path, image_path = get_radians_noise(src_path, FLAGS.pix3d_class, FLAGS.pix3d_id, np.deg2rad(FLAGS.elevation_noise), "elevation")
        # FLAGS.train_location = True
    else:
        pixel_azimuth, pixel_elevation, mesh_path, image_path = get_radians(src_path, FLAGS.pix3d_class, FLAGS.pix3d_id)
    FLAGS.init_mesh = mesh_path
    FLAGS.image = image_path
    
    
    if FLAGS.abla_ellipsoid:
        FLAGS.init_mesh = "ellipsoid"

    if FLAGS.abla_viewpoint:
        pixel_azimuth, pixel_elevation = torch.tensor(0), torch.tensor(0)
    
    AZIMUTH = -pixel_azimuth
    ELEVATION = -pixel_elevation
    RADIUS = FLAGS.radius
    
    FLAGS.images, FLAGS.ref_radii, FLAGS.ref_polars, FLAGS.ref_azimuths, FLAGS.zero123_ws = [], [], [], [], []
    FLAGS.default_zero123_w = 1

    if FLAGS.image is not None:
        if FLAGS.text is None:
            FLAGS.guidance = ['zero123']
            FLAGS.guidance_scale = 5

            FLAGS.images += [FLAGS.image]
            FLAGS.ref_radii += [FLAGS.default_radius]
            FLAGS.ref_polars += [FLAGS.default_polar]
            FLAGS.ref_azimuths += [FLAGS.default_azimuth]
            FLAGS.zero123_ws += [FLAGS.default_zero123_w]
        
        else:
            FLAGS.guidance = ['SD', 'clip']
            FLAGS.guidance_scale = 10
    
    # reset to None
    if len(FLAGS.images) == 0:
        FLAGS.images = None
    
    guidance_model = torch.nn.ModuleDict()

    if 'SD' in FLAGS.guidance:
        from guidance.sd_utils import StableDiffusion
        guidance_model['SD'] = StableDiffusion(device=device, fp16=True, vram_O=False, sd_version=FLAGS.sd_version, FLAGS=FLAGS)
    
    if 'zero123' in FLAGS.guidance:
        from guidance.zero123_utils import Zero123
        # guidance_model['zero123'] = Zero123('cuda', fp16=True)
        guidance_model['zero123'] = Zero123(device=device, fp16=True, config=FLAGS.zero123_config, ckpt=FLAGS.zero123_ckpt, vram_O=False, t_range=FLAGS.t_range, FLAGS=FLAGS)
    
    if 'clip' in FLAGS.guidance:
        from guidance.clip_utils import CLIP
        guidance_model['clip'] = CLIP(device=device)

    embeddings = EasyDict()
    
    for key in guidance_model:
        guidance_model[key].eval()
        for p in guidance_model[key].parameters():
            p.requires_grad = False
        embeddings[key] = {}
    
    text_z = []

    # ==============================================================================================
    #  Create    pipeline
    # ==============================================================================================
    
    # load azimuth and elevation for pixel loss
    # pixel_azimuth, pixel_elevation = spherical_radians("dataset/pix3d.json")
    # print(f"[azimuth]: {torch.rad2deg(pixel_azimuth)}")
    # print(f"[elevation]: {torch.rad2deg(pixel_elevation)}")
    # FLAGS.pixel_azimuth = -pixel_azimuth
    # FLAGS.pixel_elevation = -pixel_elevation

    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================
    
    if FLAGS.learn_light:
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=FLAGS.lgt_bias, FLAGS=FLAGS)
        light.save_env_map(os.path.join(FLAGS.out_dir, "initial_probe.hdr"), lgt)
    else:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale, FLAGS=FLAGS)


    if FLAGS.change_light:
        print("[INFO] change light envmap list loading...")
        env_path = glob.glob(f"data/irrmaps/rot_hdr/*.hdr")
        env_path.sort()
        FLAGS.envmap_path = env_path

    # ==============================================================================================
    #  always use DMtets to create geometry
    # ==============================================================================================

    # Setup geometry for optimization
    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)

    # Setup textures, make initial guess from reference if possible
    mat = initial_guess_material(geometry, True, FLAGS)
    # mat = initial_guess_material(geometry, True, FLAGS, bsdf='diffuse')  # FIXME: bsdf diffuse
    
    # mat['kd_ks_normal'] = MLPTexture3D
    # mat['bsdf'] = 'pbr'
    
    # wandb logging
    if FLAGS.local_rank == 0:
        # wandb
        wandb.init(project="Meta3D",
                entity=FLAGS.wandb_user,
                name=FLAGS.out_name,
                config=FLAGS,
                dir=FLAGS.out_dir,
                tags=[FLAGS.pix3d_class])
    
    # Run optimization
    # geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate, FLAGS.optim, guidance_model, text_z,
    #                 FLAGS, pass_idx=0, pass_name="dmtet_pass1", optimize_light=FLAGS.learn_light)
    # load azimuth and elevation for pixel loss
    

    print(f"[azimuth]: {torch.rad2deg(AZIMUTH)}")
    print(f"[elevation]: {torch.rad2deg(ELEVATION)}")
    print(f"[radius]: {RADIUS}")

    cam_params = EasyDict()
    cam_params.radius = RADIUS
    cam_params.azimuth = AZIMUTH
    cam_params.elevation = ELEVATION

    # parameters which will be updated in pre-optimization stages
    target_params = EasyDict()
    
    if FLAGS.optim_radius and FLAGS.optim_location:
        print("[Both radius and location optim]")
        target_params.radius = torch.tensor([RADIUS], dtype=torch.float32, requires_grad=True)
        target_params.azimuth = torch.tensor([AZIMUTH], dtype=torch.float32, requires_grad=True)
        target_params.elevation = torch.tensor([ELEVATION], dtype=torch.float32, requires_grad=True)
    
        dataset_optim = DatasetDream(glctx,  FLAGS, train=False, validate=False, direction="optim", 
                                     optim_radius=FLAGS.optim_radius, optim_location=FLAGS.optim_location,
                                     cam_params=cam_params, target_params=target_params)

        optimize_cam(glctx, geometry, mat, lgt, dataset_optim, FLAGS.optim, guidance_model, text_z, embeddings,
                FLAGS, pass_idx=0, pass_name="dmtet_pass1", optimize_light=False, target_params=target_params)
        
        cam_params.radius = target_params.radius.item()
        cam_params.azimuth = target_params.azimuth.item()
        cam_params.elevation = target_params.elevation.item()
        print(f"[Updated radius]: {cam_params.radius}")
        print(f"[Update azimuth]: {cam_params.azimuth}")
        print(f"[Update elevation]: {cam_params.elevation}")

    elif FLAGS.optim_radius:
        print("[Radius optim]")
        target_params.radius = torch.tensor([RADIUS], dtype=torch.float32, requires_grad=True)

        dataset_optim = DatasetDream(glctx,  FLAGS, train=False, validate=False, direction="optim", 
                                     optim_radius=FLAGS.optim_radius, optim_location=FLAGS.optim_location,
                                     cam_params=cam_params, target_params=target_params)

        optimize_cam(glctx, geometry, mat, lgt, dataset_optim, FLAGS.optim, guidance_model, text_z, embeddings,
                FLAGS, pass_idx=0, pass_name="dmtet_pass1", optimize_light=False, target_params=target_params)

        cam_params.radius = target_params.radius.item()
        print(f"[Updated radius]: {cam_params.radius}")

    elif FLAGS.optim_elevation:
        print("[Elevation optim]")
        target_params.azimuth = torch.tensor([AZIMUTH], dtype=torch.float32, requires_grad=False)
        target_params.elevation = torch.tensor([ELEVATION], dtype=torch.float32, requires_grad=True)

        dataset_optim = DatasetDream(glctx,  FLAGS, train=False, validate=False, direction="optim", 
                                optim_radius=FLAGS.optim_radius, optim_location=FLAGS.optim_location,
                                cam_params=cam_params, target_params=target_params)

        optimize_cam(glctx, geometry, mat, lgt, dataset_optim, FLAGS.optim, guidance_model, text_z, embeddings,
                FLAGS, pass_idx=0, pass_name="dmtet_pass1", optimize_light=False, target_params=target_params)

        cam_params.azimuth = target_params.azimuth.item()
        cam_params.elevation = target_params.elevation.item()
        print(f"[Before azimuth]: {AZIMUTH}")
        print(f"[Update azimuth]: {cam_params.azimuth}")
        print(f"[Before elevation]: {ELEVATION}")
        print(f"[Update elevation]: {cam_params.elevation}")

    elif FLAGS.optim_location:
        print("[Location optim]")
        target_params.azimuth = torch.tensor([AZIMUTH], dtype=torch.float32, requires_grad=True)
        target_params.elevation = torch.tensor([ELEVATION], dtype=torch.float32, requires_grad=True)

        dataset_optim = DatasetDream(glctx,  FLAGS, train=False, validate=False, direction="optim", 
                                optim_radius=FLAGS.optim_radius, optim_location=FLAGS.optim_location,
                                cam_params=cam_params, target_params=target_params)

        optimize_cam(glctx, geometry, mat, lgt, dataset_optim, FLAGS.optim, guidance_model, text_z, embeddings,
                FLAGS, pass_idx=0, pass_name="dmtet_pass1", optimize_light=False, target_params=target_params)

        cam_params.azimuth = target_params.azimuth.item()
        cam_params.elevation = target_params.elevation.item()
        print(f"[Update azimuth]: {cam_params.azimuth}")
        print(f"[Update elevation]: {cam_params.elevation}")
    

    
    del target_params

    # if FLAGS.optimize_dataset:
    #     target_params = EasyDict()
    #     target_params.target_radius = torch.tensor([RADIUS], dtype=torch.float32, requires_grad=True)
    #     target_params.target_azimuth = torch.tensor([AZIMUTH], dtype=torch.float32, requires_grad=True)
    #     target_params.target_elevation = torch.tensor([ELEVATION], dtype=torch.float32, requires_grad=True)

    #     dataset_optim = DatasetDream(glctx,  FLAGS, train=False, validate=False, direction="pixel", 
    #                                   cam_params=cam_params, optimize=True, target_params=target_params)

    #     optimize_cam(glctx, geometry, mat, lgt, dataset_optim, FLAGS.optim, guidance_model, text_z, embeddings,
    #             FLAGS, pass_idx=0, pass_name="dmtet_pass1", optimize_light=FLAGS.learn_light, target_params=target_params)

    #     cam_params.cam_radius = target_params.target_radius.item()
    #     print(f"Update radius: {cam_params.cam_radius}")

    datasets = EasyDict()
    target_params = EasyDict()
    print(f"[DEBUG] first radius: {cam_params.radius}")
    
    # FIXME: it is for rebuttal, have to remove later
    if FLAGS.train_azimuth:
        target_params.azimuth = torch.tensor([AZIMUTH], dtype=torch.float32, requires_grad=True)
        target_params.elevation = torch.tensor([ELEVATION], dtype=torch.float32, requires_grad=False)
        target_params.radius = torch.tensor([cam_params.radius], dtype=torch.float32, requires_grad=False)

        dataset_train    = DatasetDream(glctx, FLAGS, train=True, validate=False, direction=False, train_location=FLAGS.train_location, cam_params=target_params)
        dataset_validate = DatasetDream(glctx, FLAGS, train=False, validate=True, direction=False, train_location=FLAGS.train_location, cam_params=target_params)
        dataset_front = DatasetDream(glctx, FLAGS, train=False, validate=False, direction="front", train_location=FLAGS.train_location, cam_params=target_params)
        dataset_back = DatasetDream(glctx, FLAGS, train=False, validate=False, direction="back", train_location=FLAGS.train_location, cam_params=target_params)
        dataset_pixel = DatasetDream(glctx, FLAGS, train=False, validate=False, direction="pixel", train_location=FLAGS.train_location, cam_params=target_params)

        dataset_inference    = DatasetDream(glctx, FLAGS, train=True, validate=False, direction=False, train_location=FLAGS.train_location, cam_params=target_params)

    elif FLAGS.train_elevation:
        target_params.azimuth = torch.tensor([AZIMUTH], dtype=torch.float32, requires_grad=False)
        target_params.elevation = torch.tensor([ELEVATION], dtype=torch.float32, requires_grad=True)
        target_params.radius = torch.tensor([cam_params.radius], dtype=torch.float32, requires_grad=False)

        dataset_train    = DatasetDream(glctx, FLAGS, train=True, validate=False, direction=False, train_location=FLAGS.train_location, cam_params=target_params)
        dataset_validate = DatasetDream(glctx, FLAGS, train=False, validate=True, direction=False, train_location=FLAGS.train_location, cam_params=target_params)
        dataset_front = DatasetDream(glctx, FLAGS, train=False, validate=False, direction="front", train_location=FLAGS.train_location, cam_params=target_params)
        dataset_back = DatasetDream(glctx, FLAGS, train=False, validate=False, direction="back", train_location=FLAGS.train_location, cam_params=target_params)
        dataset_pixel = DatasetDream(glctx, FLAGS, train=False, validate=False, direction="pixel", train_location=FLAGS.train_location, cam_params=target_params)

        dataset_inference    = DatasetDream(glctx, FLAGS, train=True, validate=False, direction=False, train_location=FLAGS.train_location, cam_params=target_params)
        
    elif FLAGS.train_location:
        target_params.azimuth = torch.tensor([AZIMUTH], dtype=torch.float32, requires_grad=True)
        target_params.elevation = torch.tensor([ELEVATION], dtype=torch.float32, requires_grad=True)
        target_params.radius = torch.tensor([cam_params.radius], dtype=torch.float32, requires_grad=False)

        dataset_train    = DatasetDream(glctx, FLAGS, train=True, validate=False, direction=False, train_location=FLAGS.train_location, cam_params=target_params)
        dataset_validate = DatasetDream(glctx, FLAGS, train=False, validate=True, direction=False, train_location=FLAGS.train_location, cam_params=target_params)
        dataset_front = DatasetDream(glctx, FLAGS, train=False, validate=False, direction="front", train_location=FLAGS.train_location, cam_params=target_params)
        dataset_back = DatasetDream(glctx, FLAGS, train=False, validate=False, direction="back", train_location=FLAGS.train_location, cam_params=target_params)
        dataset_pixel = DatasetDream(glctx, FLAGS, train=False, validate=False, direction="pixel", train_location=FLAGS.train_location, cam_params=target_params)

        dataset_inference    = DatasetDream(glctx, FLAGS, train=True, validate=False, direction=False, train_location=FLAGS.train_location, cam_params=target_params)

    else:
        # TODO: check below code is correct
        target_params.azimuth = torch.tensor([AZIMUTH], dtype=torch.float32, requires_grad=False)
        target_params.elevation = torch.tensor([ELEVATION], dtype=torch.float32, requires_grad=False)
        target_params.radius = torch.tensor([cam_params.radius], dtype=torch.float32, requires_grad=False)

        dataset_train    = DatasetDream(glctx, FLAGS, train=True, validate=False, direction=False, train_location=FLAGS.train_location, cam_params=cam_params)
        dataset_validate = DatasetDream(glctx, FLAGS, train=False, validate=True, direction=False, train_location=FLAGS.train_location, cam_params=cam_params)
        dataset_front = DatasetDream(glctx, FLAGS, train=False, validate=False, direction="front", train_location=FLAGS.train_location, cam_params=cam_params)
        dataset_back = DatasetDream(glctx, FLAGS, train=False, validate=False, direction="back", train_location=FLAGS.train_location, cam_params=cam_params)
        dataset_pixel = DatasetDream(glctx, FLAGS, train=False, validate=False, direction="pixel", train_location=FLAGS.train_location, cam_params=cam_params)

        dataset_inference    = DatasetDream(glctx, FLAGS, train=True, validate=False, direction=False, train_location=FLAGS.train_location, cam_params=cam_params)
        
    datasets.train = dataset_train
    datasets.validate = dataset_validate
    datasets.front = dataset_front
    datasets.back = dataset_back
    datasets.pixel = dataset_pixel

    geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, datasets, FLAGS.optim, guidance_model, text_z, embeddings,
                FLAGS, pass_idx=0, pass_name="dmtet_pass1", optimize_light=FLAGS.learn_light, target_params=target_params)

    if FLAGS.local_rank == 0 and FLAGS.validate:
        validate(glctx, geometry, mat, lgt, dataset_validate, dataset_inference, os.path.join(FLAGS.out_dir, "dmtet_validate"), FLAGS)

    # Create textured mesh from result
    base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS)

    # Free temporaries / cached memory 
    torch.cuda.empty_cache()
    mat['kd_ks_normal'].cleanup()
    del mat['kd_ks_normal']

    lgt = lgt.clone()
    geometry = DLMesh(base_mesh, FLAGS)

    if FLAGS.local_rank == 0:
        # Dump mesh for debugging.
        os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
        light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"), lgt)

    import pickle
    if FLAGS.optim_radius and FLAGS.train_location:
        my_dict = {"azimuth": target_params.azimuth, "elevation": target_params.elevation, "radius": target_params.radius}
    elif FLAGS.optim_radius:
        my_dict = {"azimuth": cam_params.azimuth, "elevation": cam_params.elevation, "radius": target_params.radius}
    elif FLAGS.train_location:
        my_dict = {"azimuth": target_params.azimuth, "elevation": target_params.elevation, "radius": cam_params.radius}
    else:
        my_dict = {"azimuth": cam_params.azimuth, "elevation": cam_params.elevation, "radius": cam_params.radius}

    with open (f'{FLAGS.out_dir}/dmtet_validate/my_dict.pkl', 'wb') as file:
        pickle.dump(my_dict, file)
    # # ==============================================================================================
    # #  Pass 2: Train with fixed topology (mesh)
    # # ==============================================================================================
    # geometry, mat = optimize_mesh(glctx, geometry, base_mesh.material, lgt, dataset_train, dataset_validate, guidance_model, text_z, 
    #             FLAGS, pass_idx=1, pass_name="mesh_pass", warmup_iter=100, optimize_light=FLAGS.learn_light and not FLAGS.lock_light, 
    #             optimize_geometry=not FLAGS.lock_pos)

    # # ==============================================================================================
    # #  Validate
    # # ==============================================================================================
    # if FLAGS.validate and FLAGS.local_rank == 0:
    #     validate(glctx, geometry, mat, lgt, dataset_validate, guidance_model, text_z, os.path.join(FLAGS.out_dir, "validate"), FLAGS)

    # # ==============================================================================================
    # #  Dump output
    # # ==============================================================================================
    # if FLAGS.local_rank == 0:
    #     final_mesh = geometry.getMesh(mat)
    #     os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
    #     obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
    #     light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)

#----------------------------------------------------------------------------
