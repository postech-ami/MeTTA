import os

# If you have multiple GPUs, you can set the GPU to use here.
# The default is to use the first GPU, which is usually GPU 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import copy
from glob import glob
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
# from segment_anything import build_sam, SamPredictor 

# segment anything HQ
# from segment_anything import (
#     sam_model_registry,
#     sam_hq_model_registry,
#     SamPredictor
# )

from segment_anything_hq import (
    sam_model_registry,
    SamPredictor,
)

import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline


from huggingface_hub import hf_hub_download

### Omnidata

class BackgroundRemoval():
    def __init__(self, device='cuda'):

        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        image = Image.fromarray(image)

        image = self.interface([image])[0]
        image = np.array(image)

        return image

class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text


class DPT():
    def __init__(self, task='depth', device='cuda'):

        self.task = task
        self.device = device
        import sys; sys.path.append(os.getcwd())
        from dpt import DPTDepthModel

        if task == 'depth':
            path = 'pretrained/omnidata/omnidata_dpt_depth_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384')
            self.aug = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])

        else: # normal
            path = 'pretrained/omnidata/omnidata_dpt_normal_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
            self.aug = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor()
            ])

        # load model
        checkpoint = torch.load(path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)


    @torch.no_grad()
    def __call__(self, image):
        # image: np.ndarray, uint8, [H, W, 3]
        H, W = image.shape[:2]
        image = Image.fromarray(image)

        image = self.aug(image).unsqueeze(0).to(self.device)

        if self.task == 'depth':
            depth = self.model(image).clamp(0, 1)
            depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False)
            depth = depth.squeeze(1).cpu().numpy()
            return depth
        else:
            normal = self.model(image).clamp(0, 1)
            normal = F.interpolate(normal, size=(H, W), mode='bicubic', align_corners=False)
            normal = normal.cpu().numpy()
            return normal

### Grounded SAM
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model 


def grounded_sam(opt, local_image_path, class_name, groundingdino_model, sam_predictor):
    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    # ckpt_repo_id = "ShilongLiu/GroundingDINO"
    # ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    # ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    # groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    # sam_checkpoint = 'pretrained_sam/sam_vit_h_4b8939.pth'
    # sam = build_sam(checkpoint=sam_checkpoint)
    # device = "cpu"
    # sam.to(device=device)
    # sam_predictor = SamPredictor(sam)

    import io

    TEXT_PROMPT = f"{opt.text} {class_name}"
    # TEXT_PROMPT = f"The most front-facing middle {class_name}"
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(local_image_path)

    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    # Image.fromarray(image_source)
    # Image.fromarray(annotated_frame)

    # set image
    sam_predictor.set_image(image_source)

    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)

    masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )

    def show_mask(mask, image, random_color=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

    def get_mask(mask):
        h, w = mask.shape[-2:]
        mask = mask.type(torch.float64)
        mask = (mask.cpu().numpy() * 255).astype(np.uint8)
        
        # mask_image_pil = Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8))
        # mask_image_pil = mask_image_pil.convert("RGBA")

        # foreground: white (255), background: black (0)
        # return mask_image_pil
        # return np.array(mask_image_pil)
        return mask
 
    return masks[0][0].cpu().numpy()
    return get_mask(masks[0][0])
    
    annotated_frame_with_mask = show_mask(masks[0][0], annotated_frame)
    
    # RGBA -> RGB
    pil_img = Image.fromarray(annotated_frame_with_mask[..., :3])
    pil_img.save("_dino4.jpg")

    image_mask = masks[0][0].cpu().numpy()

    image_source_pil = Image.fromarray(image_source)
    annotated_frame_pil = Image.fromarray(annotated_frame)
    image_mask_pil = Image.fromarray(image_mask)
    annotated_frame_with_mask_pil = Image.fromarray(annotated_frame_with_mask)

    # image_mask_pil

def segment_image(opt, class_name, groundingdino_model, sam_predictor):
    out_dir = os.path.dirname(opt.path)
    out_rgba = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_rgba.png')
    out_depth = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_depth.png')
    out_normal = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_normal.png')
    out_caption = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_caption.txt')
    
    # load image
    print(f'[INFO] loading image...')
    print(f"[NAME] {opt.path}")
    image = cv2.imread(opt.path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # carve background
    print(f'[INFO] background removal...')
    mask = grounded_sam(opt, opt.path, class_name, groundingdino_model, sam_predictor) # [H, W]
    masked_image = (image * mask[..., None] + (255 - mask * 255)[..., None]).astype(np.uint8)
    alpha = ((mask * 255)[..., None]).astype(np.uint8)
    carved_image = np.concatenate((masked_image, alpha), axis=-1)
    # mask = carved_image[..., -1] > 0

    # predict depth
    print(f'[INFO] depth estimation...')
    dpt_depth_model = DPT(task='depth')
    depth = dpt_depth_model(image)[0]
    depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
    depth[~mask] = 0
    depth = (depth * 255).astype(np.uint8)
    del dpt_depth_model

    # predict normal
    print(f'[INFO] normal estimation...')
    dpt_normal_model = DPT(task='normal')
    normal = dpt_normal_model(image)[0]
    normal = (normal * 255).astype(np.uint8).transpose(1, 2, 0)
    normal[~mask] = 0
    del dpt_normal_model

    # recenter
    if opt.recenter:
        print(f'[INFO] recenter...')
        final_rgba = np.zeros((opt.size, opt.size, 4), dtype=np.uint8)
        final_depth = np.zeros((opt.size, opt.size), dtype=np.uint8)
        final_normal = np.zeros((opt.size, opt.size, 3), dtype=np.uint8)

        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        desired_size = int(opt.size * (1 - opt.border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (opt.size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (opt.size - w2) // 2
        y2_max = y2_min + w2
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        final_depth[x2_min:x2_max, y2_min:y2_max] = cv2.resize(depth[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        final_normal[x2_min:x2_max, y2_min:y2_max] = cv2.resize(normal[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)

    else:
        final_rgba = carved_image
        final_depth = depth
        final_normal = normal

    # write output
    cv2.imwrite(out_rgba, cv2.cvtColor(final_rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(out_depth, final_depth)
    cv2.imwrite(out_normal, final_normal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--single_image', action='store_true')
    parser.add_argument('--single_dir', action='store_true')
    parser.add_argument('--text', type=str, default='The largest')

    parser.add_argument('--size', default=256, type=int, help="output resolution")
    parser.add_argument('--border_ratio', default=0.2, type=float, help="output border ratio")
    parser.add_argument('--recenter', type=bool, default=True, help="recenter, potentially not helpful for multiview zero123")
    parser.add_argument('--dont_recenter', dest='recenter', action='store_false')
    opt = parser.parse_args()

    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    # sam_checkpoint = 'pretrained_sam/sam_vit_h_4b8939.pth'
    # sam = build_sam(checkpoint=sam_checkpoint)
    device = "cpu"
    # sam.to(device=device)
    # sam_predictor = SamPredictor(sam)
    
    # sam_hq_chekpoint = 'pretrained_sam/sam_hq_vit_h.pth'
    sam_hq_chekpoint = 'pretrained/sam/sam_hq_vit_h.pth'
    # sam_predictor = SamPredictor(sam_hq_model_registry["vit_h"](checkpoint=sam_hq_chekpoint).to(device))
    sam_predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=sam_hq_chekpoint).to(device))


    if opt.single_image:
        opt.path = opt.root
        class_name = os.path.dirname(opt.path).split("/")[-2]
        segment_image(opt, class_name, groundingdino_model, sam_predictor)
    elif opt.single_dir:
        class_name = opt.root.split()[-1]
        print(f"[{class_name}] ...")
        img_dir = os.path.join(opt.root, "image")
        img_list = glob(os.path.join(img_dir, "*"))
        for img_name in tqdm(img_list):
            opt.path = img_name
            segment_image(opt, class_name, groundingdino_model, sam_predictor)
    else:
        classes = os.listdir(opt.root)
        classes = [item for item in classes if os.path.isdir(os.path.join(opt.root, item))]
        for class_name in classes:
            print(f"[{class_name}] ...")
            img_dir = os.path.join(opt.root, class_name, "image")
            img_list = glob(os.path.join(img_dir, "*"))
            for img_name in tqdm(img_list):
                opt.path = img_name
                segment_image(opt, class_name, groundingdino_model, sam_predictor)
