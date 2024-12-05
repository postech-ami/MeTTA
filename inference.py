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
from utils.inference import *
from utils.transform_cam import *
from utils.utils import seed_everything

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
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=100)
    # parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-lr', '--learning-rate', type=float, nargs="*", default=0.001)
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

    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--dmtet_reso_scale', type=float, default=8, help="multiply --h/w by this for dmtet finetuning")
    
    parser.add_argument('--lambda_guidance', type=float, default=1, help="loss scale for SDS")
    parser.add_argument('--lambda_rgb', type=float, default=1000, help="loss scale for RGB")
    parser.add_argument('--lambda_mask', type=float, default=500, help="loss scale for mask (alpha)")
    parser.add_argument('--lambda_normal', type=float, default=0, help="loss scale for normal map")
    parser.add_argument('--lambda_depth', type=float, default=10, help="loss scale for relative depth")
    parser.add_argument('--lambda_2d_normal_smooth', type=float, default=0, help="loss scale for 2D normal image smoothness")
    parser.add_argument('--lambda_3d_normal_smooth', type=float, default=0, help="loss scale for 3D normal image smoothness")

    parser.add_argument('--lambda_mesh_normal', type=float, default=0.5, help="loss scale for mesh normal smoothness")
    parser.add_argument('--lambda_mesh_laplacian', type=float, default=0.5, help="loss scale for mesh laplacian")


    parser.add_argument('--lgt_bias', type=float, default=0.5, help="create_trainable_env_rnd bias")

    parser.add_argument('--lgt_interval', type=int, default=0)
    parser.add_argument('--mat_interval', type=int, default=0)
    parser.add_argument('--mat_stat_log', action='store_true')

    parser.add_argument('--debug', action='store_true')
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
    parser.add_argument('--train_location', action='store_true')

    # debugging
    parser.add_argument('--init_mesh_type', default="rcnn", choices=['rcnn', 'total3d'])
    parser.add_argument('--hard_time', action='store_true')
    parser.add_argument('--same_time', action='store_true')
    parser.add_argument('--init_mesh_thicker', type=float, default=0.)
    parser.add_argument('--pix3d_id', type=str, default="0169")
    parser.add_argument('--pix3d_class', type=str, default="chair")
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--data_type', type=str, default="pix3d")
    parser.add_argument('--root_path', type=str, default="/node_data/ugkim/sitto")

    # only for inference
    parser.add_argument('--infer_idx', type=int, default=2)
    parser.add_argument('--use_normal', action='store_true')
    

    parser.add_argument("--new_radius", type=float, default=2.5)

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
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
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

    RADIUS = FLAGS.new_radius
    
    FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

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

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res

    if FLAGS.out_dir is None:
        FLAGS.out_dir = '/node_data/ugkim/sitto/work_dirs/cube_%d' % (FLAGS.train_res)
    else:
        FLAGS.out_dir = '/node_data/ugkim/sitto/work_dirs/' + FLAGS.out_dir

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")


    glctx = dr.RasterizeGLContext()

    # ==============================================================================================
    #  Create Diffusion / CLIP
    # ==============================================================================================

    seed_everything(FLAGS.seed)

    if os.path.exists(os.path.join(FLAGS.out_dir, "dmtet_validate", "my_dict.pkl")):
        with open(os.path.join(FLAGS.out_dir, "dmtet_validate", "my_dict.pkl"), 'rb') as f:
            import pickle
            _my_dict = pickle.load(f)
            if isinstance(_my_dict['radius'], float):
                RADIUS = _my_dict['radius']
            else:
                RADIUS = _my_dict['radius'].item()
    
    print("[Current Radius]", RADIUS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # FIXME: add instpifu
    _path = os.path.join(FLAGS.root_path, "src", FLAGS.data_type)
    pixel_azimuth, pixel_elevation, mesh_path, image_path = get_radians(_path, FLAGS.pix3d_class, FLAGS.pix3d_id)
    FLAGS.init_mesh = mesh_path
    FLAGS.image = image_path

    FLAGS.images, FLAGS.ref_radii, FLAGS.ref_polars, FLAGS.ref_azimuths, FLAGS.zero123_ws = [], [], [], [], []
    FLAGS.default_zero123_w = 1

    
    text_z = []

    # ==============================================================================================
    #  Create    pipeline
    # ==============================================================================================

    cam_params = EasyDict()
    cam_params.radius = RADIUS
    cam_params.azimuth = 0
    cam_params.elevation = 0

    # dataset_train    = DatasetDream(glctx, RADIUS, FLAGS, validate=False)
    dataset_validate = DatasetDream(glctx, FLAGS, train=False, validate=True, direction=False, train_location=FLAGS.train_location, cam_params=cam_params)

    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================
    
    if FLAGS.learn_light:
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=FLAGS.lgt_bias, FLAGS=FLAGS)
        light.save_env_map(os.path.join(FLAGS.out_dir, "initial_probe.hdr"), lgt)
    else:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale, FLAGS=FLAGS)

    # ==============================================================================================
    #  always use DMtets to create geometry
    # ==============================================================================================

    # Setup geometry for optimization
    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS)

    # Setup textures, make initial guess from reference if possible
    mat = initial_guess_material(geometry, True, FLAGS)
    # mat['kd_ks_normal'] = MLPTexture3D
    # mat['bsdf'] = 'pbr'

    # Run optimization
    inference_mesh(glctx, geometry, mat, lgt, dataset_validate, text_z,
                    FLAGS=FLAGS, pass_idx=0, pass_name="dmtet_pass1", optimize_light=FLAGS.learn_light)

    # if FLAGS.local_rank == 0 and FLAGS.validate:
    #     validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, "dmtet_validate"), FLAGS)

    # # Create textured mesh from result
    # base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS)

    # # Free temporaries / cached memory 
    # torch.cuda.empty_cache()
    # mat['kd_ks_normal'].cleanup()
    # del mat['kd_ks_normal']

    # lgt = lgt.clone()
    # geometry = DLMesh(base_mesh, FLAGS)

    # if FLAGS.local_rank == 0:
    #     # Dump mesh for debugging.
    #     os.makedirs(os.path.join(FLAGS.out_dir, "dmtet_mesh"), exist_ok=True)
    #     obj.write_obj(os.path.join(FLAGS.out_dir, "dmtet_mesh/"), base_mesh)
    #     light.save_env_map(os.path.join(FLAGS.out_dir, "dmtet_mesh/probe.hdr"), lgt)
