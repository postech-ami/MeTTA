# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import numpy as np
# import tinycudann as tcnn
from gridencoder import GridEncoder

#######################################################################################################################################################
# Small MLP using PyTorch primitives, internal helper class
#######################################################################################################################################################

class _MLP(torch.nn.Module):
    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        
        net = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers']-1):
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=False),)

        # net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=True),)  # TODO: bias False -> True?
        self.net = torch.nn.Sequential(*net).cuda()
        
        self.net.apply(self._init_weights)
        
        # if self.loss_scale != 1.0:
        #     self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale, ))

    def forward(self, x):
        return self.net(x.to(torch.float32))

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

#######################################################################################################################################################
# Outward visible MLP class
#######################################################################################################################################################

class MLPTexture3D(torch.nn.Module):
    def __init__(self, AABB, channels = 9, internal_dims = 32, hidden = 2, min_max = None):
        super(MLPTexture3D, self).__init__()

        self.channels = channels
        self.internal_dims = internal_dims
        self.AABB = AABB
        self.min_max = min_max

        # Setup positional encoding, see https://github.com/NVlabs/tiny-cuda-nn for details
        desired_resolution = 4096
        base_grid_resolution = 16
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))

        # enc_cfg =  {
        #     "otype": "HashGrid",
        #     "n_levels": num_levels,
        #     "n_features_per_level": 2,
        #     "log2_hashmap_size": 19,
        #     "base_resolution": base_grid_resolution,
        #     "per_level_scale" : per_level_scale
	    # }

        # gradient_scaling = 128.0
        # self.encoder = tcnn.Encoding(3, enc_cfg)
        # self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))
        
        self.encoder = GridEncoder(3, num_levels, base_resolution=base_grid_resolution, per_level_scale=per_level_scale).cuda()

        # Setup MLP
        mlp_cfg = {
            "n_input_dims" : self.encoder.output_dim,
            "n_output_dims" : self.channels,
            "n_hidden_layers" : hidden,
            "n_neurons" : self.internal_dims
        }
        self.net = _MLP(mlp_cfg)
        print("Encoder output: %d dims" % (self.encoder.output_dim))

    # Sample texture at a given location
    def sample(self, texc, args=None):
        # texc: [n, h, w, 3]; [1, 512, 512, 3]
        # texc: [-1, 1] -> _texc: [0, 1]; normalize coords into [0, 1]
        _texc = (texc.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)  # torch.Size([262144, 3])
        
        # _texc[:, 0] min 0.1214, max 0.8545, mean 0.4999, std 0.1009
        # _texc[:, 1] min 0.1373, max 0.9996, mean 0.5677, std 0.1370
        # _texc[:, 2] min 0.3113, max 0.9441, mean 0.6128, std 0.1538

        p_enc = self.encoder(_texc.contiguous())
        # p_enc.shape: torch.Size([262144, 32])
        # min -0.0131, max 0.0130, mean -0.0002, std 0.0032

        # print("\nAfter encoding ====================")
        # print(f"min: {p_enc.min():.4f}, max: {p_enc.max():.4f}, mean: {p_enc.mean():.4f}, std: {p_enc.std():.4f}")
        
        out = self.net.forward(p_enc)
        # torch.Size([262144, 9])
        
        # mat_image = self.FLAGS.mat_interval and (iteration % self.FLAGS.mat_interval == 0)
        if (args != None) and (args.mat_stat_log) and (args.local_rank == 0):
            wandb_logs = args.wandb_logs
            wandb_logs.update({'Mat_before/kd/min': out[:, :3].min().item(), 'Mat_before/kd/max': out[:, :3].max().item(), 'Mat_before/kd/mean': out[:, :3].mean().item(), 'Mat_before/kd/std': out[:, :3].std().item()})
            
            wandb_logs.update({'Mat_before/ks/min': out[:, 3:6].min().item(), 'Mat_before/ks/max': out[:, 3:6].max().item(), 'Mat_before/ks/mean': out[:, 3:6].mean().item(), 'Mat_before/ks/std': out[:, 3:6].std().item()})
            wandb_logs.update({'Mat_before/roughness/min': out[:, 4].min().item(), 'Mat_before/roughness/max': out[:, 4].max().item(), 'Mat_before/roughness/mean': out[:, 4].mean().item(), 'Mat_before/roughness/std': out[:, 4].std().item()})
            wandb_logs.update({'Mat_before/metalic/min': out[:, 5].min().item(), 'Mat_before/metalic/max': out[:, 5].max().item(), 'Mat_before/metalic/mean': out[:, 5].mean().item(), 'Mat_before/metalic/std': out[:, 5].std().item()})
            
            wandb_logs.update({'Mat_before/knrm/min': out[:, 3:6].min().item(), 'Mat_before/knrm/max': out[:, 3:6].max().item(), 'Mat_before/knrm/mean': out[:, 3:6].mean().item(), 'Mat_before/knrm/std': out[:, 3:6].std().item()})
            
        # print("\Before ====================")
        # print(f"[kd] min: {out[:, :3].min():.4f}, max: {out[:, :3].max():.4f}, mean: {out[:, :3].mean():.4f}, std: {out[:, :3].std():.4f}")
        # print(f"[ks] min: {out[:, 3:6].min():.4f}, max: {out[:, 3:6].max():.4f}, mean: {out[:, 3:6].mean():.4f}, std: {out[:, 3:6].std():.4f}")
        # print(f"[knrm] min: {out[:, 6:].min():.4f}, max: {out[:, 6:].max():.4f}, mean: {out[:, 6:].mean():.4f}, std: {out[:, 6:].std():.4f}")
        
        # Sigmoid limit and scale to the allowed range
        # out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
        
        # FIXME: offset
        # out[:, :3] = out[:, :3] + 1.5
        out = torch.sigmoid(out)

        if (args != None) and (args.mat_stat_log) and (args.local_rank == 0):
            wandb_logs.update({'Mat_mid/kd/min': out[:, :3].min().item(), 'Mat_mid/kd/max': out[:, :3].max().item(), 'Mat_mid/kd/mean': out[:, :3].mean().item(), 'Mat_mid/kd/std': out[:, :3].std().item()})
            
            wandb_logs.update({'Mat_mid/ks/min': out[:, 3:6].min().item(), 'Mat_mid/ks/max': out[:, 3:6].max().item(), 'Mat_mid/ks/mean': out[:, 3:6].mean().item(), 'Mat_mid/ks/std': out[:, 3:6].std().item()})
            wandb_logs.update({'Mat_mid/roughness/min': out[:, 4].min().item(), 'Mat_mid/roughness/max': out[:, 4].max().item(), 'Mat_mid/roughness/mean': out[:, 4].mean().item(), 'Mat_mid/roughness/std': out[:, 4].std().item()})
            wandb_logs.update({'Mat_mid/metalic/min': out[:, 5].min().item(), 'Mat_mid/metalic/max': out[:, 5].max().item(), 'Mat_mid/metalic/mean': out[:, 5].mean().item(), 'Mat_mid/metalic/std': out[:, 5].std().item()})
            
            wandb_logs.update({'Mat_mid/knrm/min': out[:, 3:6].min().item(), 'Mat_mid/knrm/max': out[:, 3:6].max().item(), 'Mat_mid/knrm/mean': out[:, 3:6].mean().item(), 'Mat_mid/knrm/std': out[:, 3:6].std().item()})
            

        # print("Mid ====================")
        # print(f"[kd] min: {out[:, :3].min():.4f}, max: {out[:, :3].max():.4f}, mean: {out[:, :3].mean():.4f}, std: {out[:, :3].std():.4f}")
        # print(f"[ks] min: {out[:, 3:6].min():.4f}, max: {out[:, 3:6].max():.4f}, mean: {out[:, 3:6].mean():.4f}, std: {out[:, 3:6].std():.4f}")
        # print(f"[knrm] min: {out[:, 6:].min():.4f}, max: {out[:, 6:].max():.4f}, mean: {out[:, 6:].mean():.4f}, std: {out[:, 6:].std():.4f}")

        out = out * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]
        
        if (args != None) and (args.mat_stat_log) and (args.local_rank == 0):
            wandb_logs.update({'Mat_after/kd/min': out[:, :3].min().item(), 'Mat_after/kd/max': out[:, :3].max().item(), 'Mat_after/kd/mean': out[:, :3].mean().item(), 'Mat_after/kd/std': out[:, :3].std().item()})
            
            wandb_logs.update({'Mat_after/ks/min': out[:, 3:6].min().item(), 'Mat_after/ks/max': out[:, 3:6].max().item(), 'Mat_after/ks/mean': out[:, 3:6].mean().item(), 'Mat_after/ks/std': out[:, 3:6].std().item()})
            wandb_logs.update({'Mat_after/roughness/min': out[:, 4].min().item(), 'Mat_after/roughness/max': out[:, 4].max().item(), 'Mat_after/roughness/mean': out[:, 4].mean().item(), 'Mat_after/roughness/std': out[:, 4].std().item()})
            wandb_logs.update({'Mat_after/metalic/min': out[:, 5].min().item(), 'Mat_after/metalic/max': out[:, 5].max().item(), 'Mat_after/metalic/mean': out[:, 5].mean().item(), 'Mat_after/metalic/std': out[:, 5].std().item()})
            
            wandb_logs.update({'Mat_after/knrm/min': out[:, 3:6].min().item(), 'Mat_after/knrm/max': out[:, 3:6].max().item(), 'Mat_after/knrm/mean': out[:, 3:6].mean().item(), 'Mat_after/knrm/std': out[:, 3:6].std().item()})
            

        # FIXME: out statistics
        # _kd = out[:, :3]
        # _ks = out[:, 3:6]
        # _nrm = out[:, 6:]

        # print("After ====================")
        # print(f"[kd] min: {out[:, :3].min():.4f}, max: {out[:, :3].max():.4f}, mean: {out[:, :3].mean():.4f}, std: {out[:, :3].std():.4f}")
        # print(f"[ks] min: {out[:, 3:6].min():.4f}, max: {out[:, 3:6].max():.4f}, mean: {out[:, 3:6].mean():.4f}, std: {out[:, 3:6].std():.4f}")
        # print(f"[knrm] min: {out[:, 6:].min():.4f}, max: {out[:, 6:].max():.4f}, mean: {out[:, 6:].mean():.4f}, std: {out[:, 6:].std():.4f}")
        
        return out.view(*texc.shape[:-1], self.channels) # Remap to [n, h, w, 9]

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        pass

    def cleanup(self):
        # tcnn.free_temporary_memory()
        pass

