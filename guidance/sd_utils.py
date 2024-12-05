from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from dataclasses import dataclass

from PIL import Image
import os
import wandb

# class SpecifyGradient(torch.autograd.Function):
#     @staticmethod
#     @custom_fwd
#     def forward(ctx, input_tensor, gt_grad):
#         ctx.save_for_backward(gt_grad)
#         # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
#         return (input_tensor * gt_grad).sum()
#         # return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, grad_scale):
#         gt_grad, = ctx.saved_tensors
#         gt_grad = gt_grad * grad_scale
#         return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

@dataclass
class UNet2DConditionOutput:
    sample: torch.HalfTensor # Not sure how to check what unet_traced.pt contains, and user wants. HalfTensor or FloatTensor

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, FLAGS=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.FLAGS = FLAGS

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == 'xl':  # TODO: add stable diffusion XL version
            model_key = "stabilityai/stable-diffusion-xl-base-1.0"
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        if self.sd_version == 'xl':
            pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=precision_t)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t)

        if isfile('./unet_traced.pt'):
            # use jitted unet
            unet_traced = torch.jit.load('./unet_traced.pt')
            class TracedUNet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.in_channels = pipe.unet.in_channels
                    self.device = pipe.unet.device

                def forward(self, latent_model_input, t, encoder_hidden_states):
                    sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
                    return UNet2DConditionOutput(sample=sample)
            pipe.unet = TracedUNet()

        if is_xformers_available():
            pipe.enable_xformers_memory_efficient_attention()

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=precision_t)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.50)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt, batch=1):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        
        B, S = text_embeddings.shape[:2]
        text_embeddings = text_embeddings.repeat(1, batch, 1).view(B * batch, S, -1)

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        B, S = uncond_embeddings.shape[:2]
        uncond_embeddings = uncond_embeddings.repeat(1, batch, 1).view(B * batch, S, -1)

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=30, as_latent=False, args=None):
        grad_scale = 1  # TODO:

        if as_latent:
            # directly downsample input as latent
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)

            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if args.iteration >= 3000 and args.iteration <= 3300:
            min_step = int(self.num_train_timesteps * 0.50)
            max_step = int(self.num_train_timesteps * 0.98)
        else:
            min_step = self.min_step
            max_step = self.max_step
    
        t = torch.randint(min_step, max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            # Save input tensors for UNet
            #torch.save(latent_model_input, "train_latent_model_input.pt")
            #torch.save(t, "train_t.pt")
            #torch.save(text_embeddings, "train_text_embeddings.pt")
            # print(latent_model_input.shape, t.shape, text_embeddings.shape)
            noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
            # torch.Size([16, 4, 64, 64]) / torch.Size([16]) / torch.Size([16, 77, 1024])
        
            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # wandb visualize
        save_sds_image = args.sds_interval and (args.iteration % args.sds_interval == 0)
        
        if (args.local_rank == 0) and save_sds_image:
            wandb_logs = args.wandb_logs
            with torch.no_grad():
                self.visualize_latent(latents[[0]], "latent", wandb_logs, args)
                self.visualize_latent(noise_pred[[0]] - noise[[0]], "grad", wandb_logs, args)
                self.visualize_denoised_latent(noise[[0]], latents[[0]], text_embeddings, guidance_scale, t[[0]], "denoised_result", wandb_logs, args)
        
        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        
        # grad = w.view(-1, 1, 1, 1) * (noise_pred - noise)
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # loss = SpecifyGradient.apply(latents, grad)
        # loss = (grad * latents).sum()
        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.config.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # Save input tensors for UNet
                #torch.save(latent_model_input, "produce_latents_latent_model_input.pt")
                #torch.save(t, "produce_latents_t.pt")
                #torch.save(text_embeddings, "produce_latents_text_embeddings.pt")
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


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

        wandb_logs.update({f"Diffusion/{save_name}": wandb.Image(pil_img, caption=f"{args.iteration:07d} / shading: {args.mode}"), 'step': args.iteration})


    def visualize_denoised_latent(self, noise, latents, text_embeddings, guidance_scale, t, save_name, wandb_logs, args):
        # # debug : What image does the denoised latent draw?
        if noise is None:
            noise = torch.randn_like(latents)
        # add noise
        latents = self.scheduler.add_noise(latents, noise, t)
        # pred noise
        self.scheduler.set_timesteps(t.item())

        batch = args.batch
        B, _, _ = text_embeddings.shape

        for i, t_denoise in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t_denoise, encoder_hidden_states=text_embeddings[:int(B / batch)]).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t_denoise, latents).prev_sample
        
        imgs = self.decode_latents(latents)
        imgs = imgs.clone().detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        W, H, _ = imgs[0].shape
        pil_img = Image.fromarray(imgs[0])
        pil_img = pil_img.resize((H // 2, W // 2))
        pil_img.save(os.path.join(args.out_dir, "diffusion", save_name, f"{args.iteration:07d}.png"))

        wandb_logs.update({f"Diffusion/{save_name}": wandb.Image(pil_img, caption=f"{args.iteration:07d} / timestpe: {t.item()} / shading: {args.mode}"), 'step': args.iteration})


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1', 'xl'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    # plt.imsave("temp.jpg", imgs[0])
    plt.show()