import math
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from diffusers import DDIMScheduler

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from ldm.util import instantiate_from_config

from PIL import Image
import os
import wandb

# class SpecifyGradient(torch.autograd.Function):
#     @staticmethod
#     @custom_fwd
#     def forward(ctx, input_tensor, gt_grad):
#         ctx.save_for_backward(gt_grad)
#         # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
#         return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, grad_scale):
#         gt_grad, = ctx.saved_tensors
#         gt_grad = gt_grad * grad_scale
#         return gt_grad, None

# load model
def load_model_from_config(config, ckpt, device, vram_O=False, verbose=False):
    
    pl_sd = torch.load(ckpt, map_location='cpu')

    if 'global_step' in pl_sd and verbose:
        print(f'[INFO] Global Step: {pl_sd["global_step"]}')

    sd = pl_sd['state_dict']

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print('[INFO] missing keys: \n', m)
    if len(u) > 0 and verbose:
        print('[INFO] unexpected keys: \n', u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        if verbose:
            print('[INFO] loading EMA...')
        model.model_ema.copy_to(model.model)
        del model.model_ema
    
    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()
    
    model.eval().to(device)
    
    return model

class Zero123(nn.Module):
    def __init__(self, device, fp16,
                 config='./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml',
                 ckpt='./pretrained/zero123/zero123-xl.ckpt', vram_O=False, t_range=[0.02, 0.98],
                 FLAGS=None):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.vram_O = vram_O
        self.t_range = t_range
        self.FLAGS = FLAGS

        self.config = OmegaConf.load(config)
        # TODO: seems it cannot load into fp16...
        self.model = load_model_from_config(self.config, ckpt, device=self.device, vram_O=vram_O)

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = self.config.model.params.timesteps
        
        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            self.config.model.params.linear_start,
            self.config.model.params.linear_end,
            beta_schedule='scaled_linear',
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

    # @torch.no_grad()
    # def get_img_embeds(self, x):
    #     # x: image tensor [1, 3, 256, 256] in [0, 1]
    #     x = x * 2 - 1
    #     c = self.model.get_learned_conditioning(x) #.tile(n_samples, 1, 1)
    #     v = self.model.encode_first_stage(x).mode()
    #     return c, v
    
    # @torch.no_grad()
    # def get_img_embeds(self, x, batch=1):
    #     # x: image tensor [1, 3, 256, 256] in [0, 1]
    #     x = x * 2 - 1
        
    #     batches = []
    #     for _ in range(batch):
    #         batches.append(x)
        
    #     x = torch.cat(batches, dim=0)
        
    #     c = self.model.get_learned_conditioning(x) #.tile(n_samples, 1, 1)
    #     v = self.model.encode_first_stage(x).mode()
    #     return c, v

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor [B, 3, 256, 256] in [0, 1]
        # n_samples: multiple reference views, not batch
        
        x = x * 2 - 1
        c = [self.model.get_learned_conditioning(xx.unsqueeze(0)) for xx in x] #.tile(n_samples, 1, 1)
        v = [self.model.encode_first_stage(xx.unsqueeze(0)).mode() for xx in x]
        return c, v

    def train_step(self, embeddings, pred_rgb, polar, azimuth, radius, guidance_scale=3, as_latent=False, grad_scale=1, args=None):
        # before version, pred_rgb: tensor [1, 3, H, W] in [0, 1]
        # in my version, pred_rgb: tensor [B, 3, H, W] in [0, 1]  # FIXME: check normalized range
    
        # adjust SDS scale based on how far the novel view is from the known view
        # ref_radii = embeddings['ref_radii']
        # ref_polars = embeddings['ref_polars']
        # ref_azimuths = embeddings['ref_azimuths']

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # FIXME:
        
        if self.FLAGS.hard_time and (args.iteration >= 0) and (args.iteration <= 300):
            min_step = int(self.num_train_timesteps * 0.50)
            max_step = int(self.num_train_timesteps * 0.98)
        else:
            min_step = self.min_step
            max_step = self.max_step

        # min_step = self.min_step
        # max_step = self.max_step
        
        # TODO: maybe need weight by angle?
        grad_scale = 1.0 # claforte: I think this might converge faster...?
        
        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)  # shape torch.Size([8, 3, 512, 512])
            latents = self.encode_imgs(pred_rgb_256)  # shape torch.Size([8, 4, 32, 32])
        
        # FIXME: it samples differnt timestep at one iter
        if self.FLAGS.same_time:
            t = torch.randint(min_step, max_step + 1, (1,), dtype=torch.long, device=self.device)
            t = t.repeat(latents.shape[0])
        else:
            t = torch.randint(min_step, max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        with torch.no_grad():
            noise = torch.randn_like(latents)  # shape torch.Size([8, 4, 32, 32])
            latents_noisy = self.scheduler.add_noise(latents, noise, t)  # torch.Size([8, 4, 32, 32])

            x_in = torch.cat([latents_noisy] * 2)  # shape torch.Size([16, 4, 32, 32])
            t_in = torch.cat([t] * 2)  # shape torch.Size([16])
            
            # # Loop through each ref image
            # for (zero123_w, c_crossattn, c_concat, ref_polar, ref_azimuth, ref_radius) in zip(zero123_ws.T,
            #                                                                                   embeddings['c_crossattn'], embeddings['c_concat'],
            #                                                                                   ref_polars, ref_azimuths, ref_radii):
            
            # for multiple reference images
            c_crossattn = embeddings['c_crossattn'][0]  # shape torch.Size([1, 1, 768])
            c_concat = embeddings['c_concat'][0]  # shape torch.Size([1, 4, 32, 32])
            # ref_polar = ref_polars[0]
            # ref_azimuth = ref_azimuths[0]
            # ref_radius = ref_radii[0]
            
            T = torch.stack([torch.deg2rad(polar), torch.sin(torch.deg2rad(-azimuth)), torch.cos(torch.deg2rad(azimuth)), radius], dim=-1)  # shape torch.Size([8, 1, 4])
            # T = torch.cat([torch.deg2rad(polar), torch.sin(torch.deg2rad(-azimuth)), torch.cos(torch.deg2rad(azimuth)), radius], dim=-1)[:, None, :]
            # FIXME: multi-gpu batch version
            # T = torch.tensor([math.deg2rad(polar), math.sin(math.deg2rad(-azimuth)), math.cos(math.deg2rad(azimuth)), radius])
            # T = T[None, None, :].to(self.device)
            
            # FIXME: multi-gpu batch version
            # T = []
            # for i in range(len(polar)):
            #     _T = torch.tensor([math.radians(polar[i]), math.sin(math.radians(-azimuth[i])), math.cos(math.radians(azimuth[i])), radius])
            #     _T = _T[None, None, :].to(self.device)
            #     T.append(_T)
            # T = torch.cat(T, dim=0).to(self.device)
            
            # FIXME: before version
            # cond = {}
            # clip_emb = self.model.cc_projection(torch.cat([embeddings[0], T], dim=-1))
            # cond['c_crossattn'] = [torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)]
            # cond['c_concat'] = [torch.cat([torch.zeros_like(embeddings[1]).to(self.device), embeddings[1]], dim=0)]

            # noise_pred = self.model.apply_model(x_in, t_in, cond)

            cond = {}
            clip_emb = self.model.cc_projection(torch.cat([c_crossattn.repeat(len(T), 1, 1), T], dim=-1))  # shape torch.Size([8, 1, 768])
            cond['c_crossattn'] = [torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)]  # cond['c_crossattn'][0].shape - torch.Size([16, 1, 768])
            cond['c_concat'] = [torch.cat([torch.zeros_like(c_concat).repeat(len(T), 1, 1, 1).to(self.device), c_concat.repeat(len(T), 1, 1, 1)], dim=0)]  # cond['c_concat'][0].shape torch.Size([16, 4, 32, 32])
            noise_pred = self.model.apply_model(x_in, t_in, cond)  # shape torch.Size([16, 4, 32, 32])

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)  # split a tensor, each shape is torch.Size([8, 4, 32, 32])
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # wandb visualize
        save_sds_image = args.sds_interval and (args.iteration % args.sds_interval == 0)
        
        if (args.local_rank == 0) and save_sds_image:
            wandb_logs = args.wandb_logs
            with torch.no_grad():
                idx = 0

                def vis_debug(idx, angle):
                    args['angle'] = angle
                    self.visualize_latent(latents[[idx]], "latent", wandb_logs, args)
                    self.visualize_latent(noise_pred[[idx]] - noise[[idx]], "grad", wandb_logs, args)
                    self.visualize_denoised_latent(noise[[idx]], latents[[idx]], (c_crossattn, c_concat), polar[idx], azimuth[idx], radius[idx], guidance_scale, t[[idx]], "denoised_result", wandb_logs, args)
                # FIXME: vis_debug function is for debugging
                
                
                self.visualize_latent(latents[[idx]], "latent", wandb_logs, args)
                self.visualize_latent(noise_pred[[idx]] - noise[[idx]], "grad", wandb_logs, args)
                self.visualize_denoised_latent(noise[[idx]], latents[[idx]], (c_crossattn, c_concat), polar[idx], azimuth[idx], radius[idx], guidance_scale, t[[idx]], "denoised_result", wandb_logs, args)
        
        w = (1 - self.alphas[t])
        
        grad = (grad_scale * w)[:, None, None, None] * (noise_pred - noise)  # noise_pred, noise shpae: torch.Size([8, 4, 32, 32])
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # loss = SpecifyGradient.apply(latents, grad)
        # loss = (grad * latents).sum()
        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss
    
    # verification
    @torch.no_grad()
    def __call__(self,
            image, # image tensor [1, 3, H, W] in [0, 1]
            polar=0, azimuth=0, radius=0, # new view params
            scale=3, ddim_steps=50, ddim_eta=1, h=256, w=256, # diffusion params
            c_crossattn=None, c_concat=None, post_process=True,
        ):

        # if c_crossattn is None:
        #     embeddings = self.get_img_embeds(image)
                
        embeddings = self.get_img_embeds(image)
        
        T = torch.tensor([math.radians(polar), math.sin(math.radians(-azimuth)), math.cos(math.radians(azimuth)), radius])
        T = T[None, None, :].to(self.device)

        cond = {}
        # clip_emb = self.model.cc_projection(torch.cat([embeddings['c_crossattn'] if c_crossattn is None else c_crossattn, T], dim=-1))
        # cond['c_crossattn'] = [torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)]
        # cond['c_concat'] = [torch.cat([torch.zeros_like(embeddings['c_concat']).to(self.device), embeddings['c_concat']], dim=0)] if c_concat is None else [torch.cat([torch.zeros_like(c_concat).to(self.device), c_concat], dim=0)]
        
        clip_emb = self.model.cc_projection(torch.cat([embeddings[0][0], T], dim=-1))
        cond['c_crossattn'] = [torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)]
        cond['c_concat'] = [torch.cat([torch.zeros_like(embeddings[1][0]).to(self.device), embeddings[1][0]], dim=0)]

        # produce latents loop
        latents = torch.randn((1, 4, h // 8, w // 8), device=self.device)
        self.scheduler.set_timesteps(ddim_steps)
    
        for i, t in enumerate(self.scheduler.timesteps):
            x_in = torch.cat([latents] * 2)
            t_in = torch.cat([t.view(1)] * 2).to(self.device)

            noise_pred = self.model.apply_model(x_in, t_in, cond)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, eta=ddim_eta)['prev_sample']

        imgs = self.decode_latents(latents)
        # imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1) if post_process else imgs
        imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)
        
        return imgs

    def decode_latents(self, latents):
        # zs: [B, 4, 32, 32] Latent space image
        # with self.model.ema_scope():
        imgs = self.model.decode_first_stage(latents)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs # [B, 3, 256, 256] RGB space image    

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256] RGB space image
        # with self.model.ema_scope():
        imgs = imgs * 2 - 1
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents # [B, 4, 32, 32] Latent space image
    

    ################################################################
    ########################## visualize ##########################
    ################################################################

    def visualize_latent(self, latents, save_name, wandb_logs, args):
        _latents = latents.clone()
        imgs = self.decode_latents(_latents)
        imgs = imgs.clone().detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')
        
        W, H, _ = imgs[0].shape
        pil_img = Image.fromarray(imgs[0])
        pil_img = pil_img.resize((H // 2, W // 2))
        pil_img.save(os.path.join(args.out_dir, "diffusion", save_name, f"{args.iteration:07d}.png"))
        # pil_img.save(os.path.join(args.out_dir, "diffusion", save_name, f"{save_name}_{args.angle}.png"))

        wandb_logs.update({f"Diffusion/{save_name}": wandb.Image(pil_img, caption=f"{args.iteration:07d} / shading: {args.mode}"), 'step': args.iteration})


    def visualize_denoised_latent(self, noise, latents, embeddings, polar, azimuth, radius, guidance_scale, t, save_name, wandb_logs, args, ddim_eta=1):
        # # debug : What image does the denoised latent draw?
        if noise is None:
            noise = torch.randn_like(latents)
        # add noise
        latents = self.scheduler.add_noise(latents, noise, t)
        # pred noise
        self.scheduler.set_timesteps(t.item())

        for i, t_denoise in enumerate(self.scheduler.timesteps):
            x_in = torch.cat([latents] * 2)
            t_in = torch.cat([t_denoise.view(1)] * 2).to(self.device)
            T = torch.tensor([torch.deg2rad(polar), torch.sin(torch.deg2rad(-azimuth)), torch.cos(torch.deg2rad(azimuth)), radius])
            T = T[None, None, :].to(self.device)
            cond = {}
            clip_emb = self.model.cc_projection(torch.cat([embeddings[0], T], dim=-1))
            cond['c_crossattn'] = [torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)]
            cond['c_concat'] = [torch.cat([torch.zeros_like(embeddings[1]).to(self.device), embeddings[1]], dim=0)]

            with torch.no_grad():
                noise_pred = self.model.apply_model(x_in, t_in, cond)
                noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
                
                latents = self.scheduler.step(noise_pred, t_denoise, latents, eta=ddim_eta)['prev_sample']
        
        imgs = self.decode_latents(latents)
        imgs = imgs.clone().detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        W, H, _ = imgs[0].shape
        pil_img = Image.fromarray(imgs[0])
        pil_img = pil_img.resize((H // 2, W // 2))
        pil_img.save(os.path.join(args.out_dir, "diffusion", save_name, f"{args.iteration:07d}.png"))
        # pil_img.save(os.path.join(args.out_dir, "diffusion", save_name, f"denoised_{args.angle}.png"))

        wandb_logs.update({f"Diffusion/{save_name}": wandb.Image(pil_img, caption=f"{args.iteration:07d} / timestpe: {t.item()} / shading: {args.mode} / azimuth: {round(azimuth.item())} / polar: {round(polar.item())}"), 'step': args.iteration})
    
if __name__ == '__main__':
    import cv2
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str)
    parser.add_argument('--fp16', action='store_true', help="use float16 for training") # no use now, can only run in fp32

    parser.add_argument('--polar', type=float, default=0, help='delta polar angle in [-90, 90]')
    parser.add_argument('--azimuth', type=float, default=0, help='delta azimuth angle in [-180, 180]')
    parser.add_argument('--radius', type=float, default=0, help='delta camera radius multiplier in [-0.5, 0.5]')

    opt = parser.parse_args()

    device = torch.device('cuda')

    print(f'[INFO] loading image from {opt.input} ...')
    image = cv2.imread(opt.input, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

    print(f'[INFO] loading model ...')
    zero123 = Zero123(device, opt.fp16)

    print(f'[INFO] running model ...')
    
    dir_path = "temp_output/quokka_wb_2"

    import os
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    outputs = zero123(image, polar=0, azimuth=0, radius=0.5); plt.imsave(f"{dir_path}/radius_0.5.jpg", outputs[0])
    outputs = zero123(image, polar=0, azimuth=0, radius=0); plt.imsave(f"{dir_path}/radius_0.jpg", outputs[0])
    outputs = zero123(image, polar=0, azimuth=0, radius=-0.5); plt.imsave(f"{dir_path}/radius_-0.5.jpg", outputs[0])
    
    # outputs = zero123(image, polar=90, azimuth=0, radius=opt.radius); plt.imsave(f"{dir_path}/polar_90.jpg", outputs[0])
    # outputs = zero123(image, polar=45, azimuth=0, radius=opt.radius); plt.imsave(f"{dir_path}/polar_45.jpg", outputs[0])
    # outputs = zero123(image, polar=-45, azimuth=0, radius=opt.radius); plt.imsave(f"{dir_path}/polar_-45.jpg", outputs[0])
    # outputs = zero123(image, polar=-90, azimuth=0, radius=opt.radius); plt.imsave(f"{dir_path}/polar_-90.jpg", outputs[0])
    
    # outputs = zero123(image, polar=0, azimuth=0, radius=opt.radius); plt.imsave(f"{dir_path}/azimuth_0.jpg", outputs[0])
    # outputs = zero123(image, polar=0, azimuth=90, radius=opt.radius); plt.imsave(f"{dir_path}/azimuth_90.jpg", outputs[0])
    # outputs = zero123(image, polar=0, azimuth=180, radius=opt.radius); plt.imsave(f"{dir_path}/azimuth_180.jpg", outputs[0])
    # outputs = zero123(image, polar=0, azimuth=-90, radius=opt.radius); plt.imsave(f"{dir_path}/azimuth_-90.jpg", outputs[0])
    # outputs = zero123(image, polar=0, azimuth=-180, radius=opt.radius); plt.imsave(f"{dir_path}/azimuth_-180.jpg", outputs[0])
    # outputs = zero123(image, polar=0, azimuth=270, radius=opt.radius); plt.imsave(f"{dir_path}/azimuth_270.jpg", outputs[0])
    
    # outputs = zero123(image, polar=opt.polar, azimuth=opt.azimuth, radius=opt.radius)
    # plt.imshow(outputs[0])
    # plt.show()