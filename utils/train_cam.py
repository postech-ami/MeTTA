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
import torchvision

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
# Main shape fitter function / optimization loop
###############################################################################

class SimpleTrainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, image_loss_fn, guidance_model, text_z, embeddings, FLAGS, target_params):
        super(SimpleTrainer, self).__init__()

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

        self.params = list(self.target_params.values())

        self.prepare_embeddings(FLAGS)

    def forward(self, target, it, args):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])

        buffers, mesh, pred_rgb, pred_ws, all_edges, as_latent, iteration, t_iter = self.geometry.tick(self.glctx, target, self.light, self.material, it, False, args)

        losses = dict()
       
        # gt_rgb = self.rgb  # [1, 3, 512, 512]
        gt_mask = self.mask  # [1, 512, 512]
       
        # color loss
        # gt_rgb = gt_rgb * gt_mask[:, None].float() + (1 - gt_mask[:, None].float())  # white bg
        # rgb_loss = F.mse_loss(pred_rgb, gt_rgb)

        if (it % 100 == 0):
            if not os.path.exists(f"{self.FLAGS.out_dir}/cam_optim"):
                os.makedirs(f"{self.FLAGS.out_dir}/cam_optim")
        
            from utils.utils import save_img
            save_img(pred_ws[:, 0], f"{self.FLAGS.out_dir}/cam_optim/{it:06d}_pred_ws")
            
            if it == 0:
                save_img(gt_mask.float(), f"{self.FLAGS.out_dir}/cam_optim/000000_gt_mask")
        # mask loss
        # _loss_fn = torch.nn.MSELoss()
        _loss_fn = torch.nn.L1Loss()
        # _loss_fn = torch.nn.HuberLoss()
        # _loss_fn_center = torch.nn.MSELoss()
        # _loss_fn = torch.nn.BCELoss()
        
        # mask_loss = F.mse_loss(pred_ws[:, 0], gt_mask.float())
        
        mask_loss = _loss_fn(pred_ws[:, 0], gt_mask.float())
        losses['mask'] = mask_loss
        
        # TODO: resize mask to smaller size
        # _pred_ws = pred_ws[:, 0]
        # _gt_mask = gt_mask.float()

        # torch_resize = torchvision.transforms.Resize((20, 20))
        # _pred_ws_resize = torch_resize(_pred_ws)
        # _gt_mask_resize = torch_resize(_gt_mask)

        # mask_loss = _loss_fn(_pred_ws_resize, _gt_mask_resize)
        
        # losses['mask'] = mask_loss

        # TODO: center loss, bbox loss
        # _pred_ws = pred_ws[:, 0]
        # _gt_mask = gt_mask.float()

        # _, H, W = _gt_mask.shape
        # (x1, y1, x2, y2)
        # center
        # pred_bbox = torchvision.ops.masks_to_boxes(_pred_ws)[0]
        # pred_x = (pred_bbox[0] + pred_bbox[2]) / 2
        # pred_y = (pred_bbox[1] + pred_bbox[3]) / 2

        # gt_bbox = torchvision.ops.masks_to_boxes(_gt_mask)[0]
        # gt_x = (gt_bbox[0] + gt_bbox[2]) / 2
        # gt_y = (gt_bbox[1] + gt_bbox[3]) / 2

        # bbox range
        # _pred_ws = _pred_ws.squeeze(0)
        # pred_rows = torch.any(_pred_ws, dim=1)
        # pred_cols = torch.any(_pred_ws, dim=0)
        # pred_ymin, pred_ymax = torch.where(pred_rows)[0][[0, -1]]
        # pred_xmin, pred_xmax = torch.where(pred_cols)[0][[0, -1]]
        # pred_area = (pred_ymax - pred_ymin) * (pred_xmax - pred_xmin)

        # _gt_mask = _gt_mask.squeeze(0)
        # gt_rows = torch.any(_gt_mask, dim=1)
        # gt_cols = torch.any(_gt_mask, dim=0)
        # gt_ymin, gt_ymax = torch.where(gt_rows)[0][[0, -1]]
        # gt_xmin, gt_xmax = torch.where(gt_cols)[0][[0, -1]]
        # gt_area = (gt_ymax - gt_ymin) * (gt_xmax - gt_xmin)

        # range loss
        # mask_loss = _loss_fn(_pred_ws.sum(), _gt_mask.sum())

        # losses['mask'] = mask_loss

        # x_loss = _loss_fn(pred_x / W, gt_x / W)
        # y_loss = _loss_fn(pred_y / H, gt_y / H)
        # losses['x'] = x_loss
        # losses['y'] = y_loss

        return losses
    

    # calculate the text embs.
    @torch.no_grad()
    def prepare_embeddings(self, args):
        if args.images is not None:
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

def early_stopping():
    pass

def optimize_cam(
    glctx,
    geometry,
    opt_material,
    lgt,
    dataset,
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

    image_loss_fn = createLoss(FLAGS)
    trainer_noddp = SimpleTrainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, image_loss_fn, guidance, text_z, embeddings, FLAGS, target_params)

    # load latest model
    load_it = trainer_noddp.load()
    if FLAGS.multi_gpu: 
        # Multi GPU training mode
        from torch.nn.parallel import DistributedDataParallel as DDP
        trainer = DDP(trainer_noddp, device_ids=[FLAGS.local_rank], output_device=FLAGS.local_rank, find_unused_parameters=True)
        trainer.train()

    else:
        # Single GPU training mode
        trainer = trainer_noddp
        trainer.train()
    # FIXME: change lr
    _lr = 5e-3
    if optim == 'adan':
        from optimizer import Adan
        # FIXME: change lr, default 5e-3
        optimizer = Adan(trainer_noddp.params, lr=_lr, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(trainer_noddp.params, lr=_lr)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.collate)

    temp_iter = []
    temp_loss = []

    # early stopping
    stop_prev_loss = np.inf
    stop_cur_loss = 0.0

    # for it, target in enumerate(dataloader):
    for it, target in enumerate(tqdm(dataloader, desc="[pre-optimize]")):
        # load checkpoints
        it += load_it
        if it > FLAGS.pre_iter:
            break
    
        # Mix randomized background into dataset image
        target = prepare_batch(target, 'random')
        
        optimizer.zero_grad()

        # TODO: args
        args = EasyDict()
        args.iteration = it
        args.wandb_logs = None
        args.update(vars(FLAGS))
        
        losses = trainer(target, it, args)

        # loss = losses['mask'] + losses['x'] + losses['y'] * 100
        loss = losses['mask']
        loss.backward()
        optimizer.step()
        stop_cur_loss += loss.item()

        if it % 10 == 0:
            # print(f"[pre-optimize] iters: {it}, mask loss: {losses['mask']}, x loss: {losses['x']}, y loss: {losses['y'] * 100}")        
            print(f"[pre-optimize] iters: {it}, mask loss: {loss}, params: {trainer_noddp.params}")        
            temp_iter.append(it)
            temp_loss.append(loss)

        if (it != 0) and (it % 500 == 0):
            if stop_prev_loss < stop_cur_loss:
                break
            else:
                stop_prev_loss = stop_cur_loss
                stop_cur_loss = 0.0

    temp_loss = [t.item() for t in temp_loss]
    
    import matplotlib.pyplot as plt
    plt.plot(temp_iter, temp_loss)
    plt.xlabel("Iters")
    plt.ylabel("Loss")
    plt.savefig(f"{FLAGS.out_dir}/cam_optim/cam-optim")

    import pickle
    my_cam_dict = {'iter': temp_iter, 'loss': temp_loss}
    with open (f'{FLAGS.out_dir}/cam_optim/my_cam_dict.pkl', 'wb') as file:
        pickle.dump(my_cam_dict, file)

    del dataloader
    del trainer
    del trainer_noddp

