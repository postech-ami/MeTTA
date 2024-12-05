# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch

from render import util
from render import mesh
from render import render
from render import light

from .dataset import Dataset

###############################################################################
# Reference dataset using mesh & rendering
###############################################################################

def rad2deg(angle):
    return angle / np.pi * 180

def deg2rad(angle):
    return angle / 180 * np.pi

class DatasetDream(Dataset):

    def __init__(self, glctx, FLAGS, train=True, validate=False, direction=False, 
                 optim_radius=False, optim_location=False, train_location=False,
                 cam_params=None, target_params=None):
        # Init 
        self.glctx              = glctx
        self.FLAGS              = FLAGS
        self.train              = train
        self.validate           = validate
        self.direction          = direction
        self.aspect             = FLAGS.train_res[1] / FLAGS.train_res[0]
        self.default_radius     = 2.5
        self.fovy               = np.deg2rad(45)
        self.aspect             = FLAGS.train_res[1] / FLAGS.train_res[0]

        self.cam_radius         = cam_params["radius"]
        self.cam_azimuth        = cam_params["azimuth"]
        self.cam_elevation      = cam_params["elevation"]

        self.optim_radius       = optim_radius
        self.optim_location     = optim_location
        self.train_location     = train_location
        

        if self.optim_radius and self.optim_location:
            self.target_radius      = target_params["radius"]
            self.target_azimuth     = target_params["azimuth"]
            self.target_elevation   = target_params["elevation"]
        elif self.optim_radius:
            self.target_radius      = target_params["radius"]
        elif self.optim_location:
            self.target_azimuth     = target_params["azimuth"]
            self.target_elevation   = target_params["elevation"]
        # angle_x: elevation (before transform to polar)
        # angle_y: azimuth

    def _rotate_scene(self, itr, ref_angle_y=0, ref_angle_x=0):
        # self.fov = np.deg2rad(45)
        proj_mtx = util.perspective(self.fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display.
        angle_y    = (itr / 50) * np.pi * 2

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        if angle_y >= 0 and angle_y <= np.pi/8 or angle_y > 15*np.pi/8:
            direction = 0
        elif angle_y > np.pi/8 and angle_y <= 7*np.pi/8:
            direction = 1
        elif angle_y > 7*np.pi/8 and angle_y <= 9*np.pi/8:
            direction = 2
        elif angle_y > 9*np.pi/8 and angle_y <= 15*np.pi/8:
            direction = 3

        angle_x = -0.4   # FIXME: for inference stage
        # mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(angle_x + ref_angle_x) @ util.rotate_y(angle_y + ref_angle_y)) # NOTE: add reference view
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        radius = self.cam_radius
        azimuth = torch.FloatTensor([angle_y])
        elev = torch.FloatTensor([angle_x])
        radius = torch.FloatTensor([self.cam_radius])

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.train_res, self.FLAGS.spp, direction, azimuth[None, ...].cuda(), elev[None, ...].cuda(), radius[None, ...].cuda()

    def _random_scene(self, ref_angle_y=0, ref_angle_x=0):
        # ==============================================================================================
        #  Setup projection matrix
        # ==============================================================================================
        iter_res = self.FLAGS.train_res
        # self.fov = np.random.uniform(np.pi/7, np.pi/4)
        proj_mtx = util.perspective(self.fovy, iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # ==============================================================================================
        #  Random camera & light position
        # ==============================================================================================

        # Random rotation/translation matrix for optimization.

        # mv     = util.translate(0, 0, -self.cam_radius) @ util.random_rotation()
        # FIXME:
        # angle_x = np.random.uniform(-np.pi/4, np.pi/18)  # -45 ~ 10
        angle_x = np.random.uniform(-np.pi/4, np.pi/4)  # -45 ~ 45  # wide elevation
        angle_y = np.random.uniform(0, 2 * np.pi)  # 0 ~ 360

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        if angle_y >= 0 and angle_y <= np.pi/8 or angle_y > 15*np.pi/8:
            direction = 0
        elif angle_y > np.pi/8 and angle_y <= 7*np.pi/8:
            direction = 1
        elif angle_y > 7*np.pi/8 and angle_y <= 9*np.pi/8:
            direction = 2
        elif angle_y > 9*np.pi/8 and angle_y <= 15*np.pi/8:
            direction = 3

        # mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(angle_x + ref_angle_x) @ util.rotate_y(angle_y + ref_angle_y))  # NOTE: add reference view
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        azimuth = torch.FloatTensor([angle_y])
        elev = torch.FloatTensor([angle_x])
        radius = torch.FloatTensor([self.cam_radius])
        
        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), iter_res, self.FLAGS.spp, direction, azimuth[None, ...].cuda(), elev[None, ...].cuda(), radius[None, ...].cuda() # Add batch dimension

    def _fixed_scene(self, ref_angle_y, ref_angle_x):
        # angle_y: azimuth
        # angle_x: elevation

        # self.fov = np.deg2rad(45)
        proj_mtx = util.perspective(self.fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Front view for display along the iterations
        # angle_y = angle_y

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        direction = 0

        # angle_x = np.random.uniform(-np.pi/4, np.pi/18)  # -45 ~ 10
        # angle_x = -0.4
        # angle_x = angle_x
        
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(ref_angle_x) @ util.rotate_y(ref_angle_y))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        azimuth = torch.FloatTensor([ref_angle_y])
        elev = torch.FloatTensor([ref_angle_x])
        radius = torch.FloatTensor([self.cam_radius])

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.train_res, self.FLAGS.spp, direction, azimuth[None, ...].cuda(), elev[None, ...].cuda(), radius[None, ...].cuda()
    
    ###################################
    ### Train azimuth and elevation ###
    ###################################
    def _rotate_scene_train(self, itr, ref_angle_y=0, ref_angle_x=0):
        # self.fov = np.deg2rad(45)
        proj_mtx = util.perspective(self.fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display.
        angle_y    = (itr / 50) * np.pi * 2

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        if angle_y >= 0 and angle_y <= np.pi/8 or angle_y > 15*np.pi/8:
            direction = 0
        elif angle_y > np.pi/8 and angle_y <= 7*np.pi/8:
            direction = 1
        elif angle_y > 7*np.pi/8 and angle_y <= 9*np.pi/8:
            direction = 2
        elif angle_y > 9*np.pi/8 and angle_y <= 15*np.pi/8:
            direction = 3

        angle_x = -0.1
        
        angle_y = torch.tensor([angle_y], dtype=torch.float32)
        angle_x = torch.tensor([angle_x], dtype=torch.float32)

        # mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
        mv     = util.translate(0, 0, -self.cam_radius) @ (util._rotate_x(angle_x + ref_angle_x) @ util._rotate_y(angle_y + ref_angle_y)) # NOTE: add reference view
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        azimuth = angle_y
        elev = angle_x
        radius = torch.FloatTensor([self.cam_radius])

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.train_res, self.FLAGS.spp, direction, azimuth[None, ...].cuda(), elev[None, ...].cuda(), radius[None, ...].cuda()

    def _random_scene_train(self, ref_angle_y=0, ref_angle_x=0):
        # ==============================================================================================
        #  Setup projection matrix
        # ==============================================================================================
        iter_res = self.FLAGS.train_res
        # self.fov = np.random.uniform(np.pi/7, np.pi/4)
        proj_mtx = util.perspective(self.fovy, iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # ==============================================================================================
        #  Random camera & light position
        # ==============================================================================================

        # Random rotation/translation matrix for optimization.

        # mv     = util.translate(0, 0, -self.cam_radius) @ util.random_rotation()
        # FIXME:
        # angle_x = np.random.uniform(-np.pi/4, np.pi/18)  # -45 ~ 10
        angle_x = np.random.uniform(-np.pi/4, np.pi/4)  # -45 ~ 45  # wide elevation
        angle_y = np.random.uniform(0, 2 * np.pi)  # 0 ~ 360

        angle_y = torch.tensor([angle_y], dtype=torch.float32)
        angle_x = torch.tensor([angle_x], dtype=torch.float32)

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        if angle_y >= 0 and angle_y <= np.pi/8 or angle_y > 15*np.pi/8:
            direction = 0
        elif angle_y > np.pi/8 and angle_y <= 7*np.pi/8:
            direction = 1
        elif angle_y > 7*np.pi/8 and angle_y <= 9*np.pi/8:
            direction = 2
        elif angle_y > 9*np.pi/8 and angle_y <= 15*np.pi/8:
            direction = 3

        # mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
        mv     = util.translate(0, 0, -self.cam_radius) @ (util._rotate_x(angle_x + ref_angle_x) @ util._rotate_y(angle_y + ref_angle_y))  # NOTE: add reference view
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        azimuth = angle_y
        elev = angle_x
        radius = torch.FloatTensor([self.cam_radius])
        
        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), iter_res, self.FLAGS.spp, direction, azimuth[None, ...].cuda(), elev[None, ...].cuda(), radius[None, ...].cuda() # Add batch dimension

    def _fixed_scene_train(self, ref_angle_y, ref_angle_x):
        # angle_y: azimuth
        # angle_x: elevation

        # self.fov = np.deg2rad(45)
        proj_mtx = util.perspective(self.fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Front view for display along the iterations
        # angle_y = angle_y

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        direction = 0

        # angle_x = np.random.uniform(-np.pi/4, np.pi/18)  # -45 ~ 10
        # angle_x = -0.4
        # angle_x = angle_x
        
        mv     = util.translate(0, 0, -self.cam_radius) @ (util._rotate_x(ref_angle_x) @ util._rotate_y(ref_angle_y))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        azimuth = ref_angle_y
        elev = ref_angle_x
        radius = torch.FloatTensor([self.cam_radius])

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.train_res, self.FLAGS.spp, direction, azimuth[None, ...].cuda(), elev[None, ...].cuda(), radius[None, ...].cuda()

    def _optim_scene(self, ref_angle_y, ref_angle_x):
        # angle_y: azimuth
        # angle_x: elevation

        # self.fov = np.deg2rad(45)
        proj_mtx = util.perspective(self.fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        direction = 0

        mv     = util._translate(self.target_radius) @ (util._rotate_x(ref_angle_x) @ util._rotate_y(ref_angle_y))
        
        azimuth = ref_angle_y
        elev = ref_angle_x
        radius = self.target_radius
            
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.train_res, self.FLAGS.spp, direction, azimuth[None, ...].cuda(), elev[None, ...].cuda(), radius[None, ...].cuda()


    def _optim_location(self, ref_angle_y, ref_angle_x):
        # angle_y: azimuth
        # angle_x: elevation

        # self.fov = np.deg2rad(45)
        proj_mtx = util.perspective(self.fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        direction = 0

        mv     = util.translate(0, 0, -self.cam_radius) @ (util._rotate_x(ref_angle_x) @ util._rotate_y(ref_angle_y))

        azimuth = ref_angle_y
        elev = ref_angle_x
        radius = torch.FloatTensor([self.cam_radius])

        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.train_res, self.FLAGS.spp, direction, azimuth[None, ...].cuda(), elev[None, ...].cuda(), radius[None, ...].cuda()


    def _optim_radius(self, ref_angle_y, ref_angle_x):
        # angle_y: azimuth
        # angle_x: elevation

        # self.fov = np.deg2rad(45)
        proj_mtx = util.perspective(self.fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        direction = 0

        azimuth = torch.FloatTensor([ref_angle_y])
        elev = torch.FloatTensor([ref_angle_x])
        
        mv     = util._translate(self.target_radius) @ (util.rotate_x(ref_angle_x) @ util.rotate_y(ref_angle_y))
        radius = self.target_radius

        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.train_res, self.FLAGS.spp, direction, azimuth[None, ...].cuda(), elev[None, ...].cuda(), radius[None, ...].cuda()


    def _debug_scene(self, itr, ref_angle_y=0, ref_angle_x=0):
        # self.fov = np.deg2rad(45)
        proj_mtx = util.perspective(self.fovy, self.FLAGS.train_res[1] / self.FLAGS.train_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display.
        # angle_y    = (itr / 4) * np.pi * 2  # TODO: only front 4 value are available
        angle_y = [0, np.pi / 2, np.pi, np.pi * (3/2), np.pi * 2]
        angle_y = angle_y[int(itr % 5)]
        # angle_y = 0

        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        if angle_y >= 0 and angle_y <= np.pi/8 or angle_y > 15*np.pi/8:
            direction = 0
        elif angle_y > np.pi/8 and angle_y <= 7*np.pi/8:
            direction = 1
        elif angle_y > 7*np.pi/8 and angle_y <= 9*np.pi/8:
            direction = 2
        elif angle_y > 9*np.pi/8 and angle_y <= 15*np.pi/8:
            direction = 3

        # angle_x = np.random.uniform(-np.pi/4, np.pi/18)  # -45 ~ 10
        angle_x = 0
        # angle_x = [-np.pi / 2, -np.pi / 4, np.pi / 4, np.pi / 2]
        # angle_x = angle_x[int(itr % 4)]
        
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        radius = self.cam_radius
        azimuth = torch.FloatTensor([angle_y])
        elev = torch.FloatTensor([angle_x])
        radius = torch.FloatTensor([self.cam_radius])

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.train_res, self.FLAGS.spp, direction, azimuth[None, ...].cuda(), elev[None, ...].cuda(), radius[None, ...].cuda()


    def __len__(self):
        if self.validate:
            return 50
        elif (self.optim_radius or self.optim_location):
            return self.FLAGS.pre_iter    
        else:
            return (self.FLAGS.iter + 1) * self.FLAGS.batch

        # return 50 if self.validate else (self.FLAGS.iter + 1) * self.FLAGS.batch

    def __getitem__(self, itr):
        # ==============================================================================================
        #  Randomize scene parameters
        # ==============================================================================================

        if self.train and self.FLAGS.train_location:
            mv, mvp, campos, iter_res, iter_spp, direction, azimuth, elev, radius = self._random_scene_train(ref_angle_y=self.cam_azimuth, ref_angle_x=self.cam_elevation)
        elif self.train:
            mv, mvp, campos, iter_res, iter_spp, direction, azimuth, elev, radius = self._random_scene(ref_angle_y=self.cam_azimuth, ref_angle_x=self.cam_elevation)
        
        elif self.validate and self.FLAGS.train_location:
            mv, mvp, campos, iter_res, iter_spp, direction, azimuth, elev, radius = self._rotate_scene_train(itr, ref_angle_y=self.cam_azimuth, ref_angle_x=self.cam_elevation)
        elif self.validate:
            mv, mvp, campos, iter_res, iter_spp, direction, azimuth, elev, radius = self._rotate_scene(itr, ref_angle_y=self.cam_azimuth, ref_angle_x=self.cam_elevation)

        elif self.direction == "pixel" and self.FLAGS.train_location:
            mv, mvp, campos, iter_res, iter_spp, direction, azimuth, elev, radius = self._fixed_scene_train(ref_angle_y=self.cam_azimuth, ref_angle_x=self.cam_elevation)
        elif self.direction == "pixel":
            mv, mvp, campos, iter_res, iter_spp, direction, azimuth, elev, radius = self._fixed_scene(ref_angle_y=self.cam_azimuth, ref_angle_x=self.cam_elevation)

        elif self.direction == "front":
            mv, mvp, campos, iter_res, iter_spp, direction, azimuth, elev, radius = self._fixed_scene(ref_angle_y=0, ref_angle_x=0)  # TODO: can add ref_angle_x=-0.4 for upper front
        elif self.direction == "back":
            mv, mvp, campos, iter_res, iter_spp, direction, azimuth, elev, radius = self._fixed_scene(ref_angle_y=np.pi, ref_angle_x=0)
                
        elif self.direction == "optim" and self.optim_location and self.optim_radius:
            mv, mvp, campos, iter_res, iter_spp, direction, azimuth, elev, radius = self._optim_scene(ref_angle_y=self.target_azimuth, ref_angle_x=self.target_elevation)
        elif self.direction == "optim" and self.optim_location:
            mv, mvp, campos, iter_res, iter_spp, direction, azimuth, elev, radius = self._optim_location(ref_angle_y=self.target_azimuth, ref_angle_x=self.target_elevation)
        elif self.direction == "optim" and self.optim_radius:
            mv, mvp, campos, iter_res, iter_spp, direction, azimuth, elev, radius = self._optim_radius(ref_angle_y=self.cam_azimuth, ref_angle_x=self.cam_elevation)
        

        elev = rad2deg(elev)
        polar = 90 + elev  # elevation to polar, range from [90, -90] to [0, 180]
        azimuth = rad2deg(azimuth)

        # delta polar/azimuth/radius to default view
        delta_polar = polar - self.FLAGS.default_polar
        delta_azimuth = azimuth - self.FLAGS.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        # delta_azimuth = -delta_azimuth  # for matching zero123 coordinates

        delta_radius = radius - self.default_radius
        
        # FIXME: split radius and delta_radius / and when I optimize radius I only optimize radius / and delta_radius should be fixed at zero123 stage
        if self.direction == "optim":
            delta_radius = radius
        else:
            # FIXME: because I optimized radius to my coordinate systems, and just use this with 0 delta radius
            delta_radius = torch.FloatTensor([0])
            delta_radius = delta_radius[None, ...].cuda()
        
        # if self.optim_radius and (itr % 10 == 0):
        #     print(f"[target radius]: {self.target_radius.item()}")
        #     print(f"[before radius]: {self.cam_radius}")
        # if self.optim_location and (itr % 10 == 0):
        #     print(f"[target azimuth]: {self.target_azimuth.item()}")
        #     print(f"[before azimuth]: {self.cam_azimuth}")
        #     print("---")
        #     print(f"[target elevation]: {self.target_elevation.item()}")
        #     print(f"[before elevation]: {self.cam_elevation}")
        
        # if self.train_location:
        #     if self.train:
        #         print("[train]===")
        #     elif self.validate:
        #         print("[validate]===")
        #     elif self.direction == "pixel":
        #         print("[pixel]==")
            
        #     print(f"azimuth: {self.cam_azimuth.item()}")
        #     print(f"elevation: {self.cam_elevation.item()}")
            
        # return angle is degree
        return {
            'mv' : mv, # [1, 4, 4], world2cam
            'mvp' : mvp, # [1, 4, 4], world2cam + projection
            'campos' : campos, # [1, 3], camera's position in world coord
            'resolution' : iter_res, # [2], training res, e.g., [512, 512]
            'spp' : iter_spp, # [1], mostly == 1
            'direction' : direction,
            'polar': delta_polar,  # [1, 1]
            'azimuth': delta_azimuth,  # [1, 1]
            'radius': delta_radius, # [1, 1],
        }

