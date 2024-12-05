import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import argparse
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from segment_anything_hq import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from typing import Any, List, Tuple, Union

# import sys; sys.path.append(os.path.dirname(os.getcwd()))
import sys; sys.path.append(os.getcwd())

from utils import EasyDict

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

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

### SAM-HQ
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    # pos_points = coords[labels==1]
    # neg_points = coords[labels==0]
    # FIXME:
    pos_points = coords
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    # ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        
        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()


def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename +'.png',bbox_inches='tight',pad_inches=-0.1)
    plt.close()


def segment_module(image_path):
    ### Omnidata
    opt = EasyDict()

    opt.path = image_path
    opt.size = 256
    opt.border_ratio = 0.2
    opt.recenter = False

    opt.depth = True
    opt.normal = True


    out_dir = os.path.dirname(opt.path)
    out_rgba = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_rgba.png')
    out_depth = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_depth.png')
    out_normal = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_normal.png')
    out_caption = os.path.join(out_dir, os.path.basename(opt.path).split('.')[0] + '_caption.txt')

    # load image
    print(f"[INFO] loading image...")
    image = cv2.imread(opt.path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # carve background
    print(f"[INFO] background removal...")
    carved_image = BackgroundRemoval()(image)  # [H, W, 4]
    mask = carved_image[..., -1] > 0

    final_rgba = carved_image

    # predict depth
    if opt.depth:
        print(f'[INFO] depth estimation...')
        dpt_depth_model = DPT(task='depth')
        depth = dpt_depth_model(image)[0]
        depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
        depth[~mask] = 0
        depth = (depth * 255).astype(np.uint8)
        del dpt_depth_model

        final_depth = depth

    # predict normal
    if opt.normal:
        print(f'[INFO] normal estimation...')
        dpt_normal_model = DPT(task='normal')
        normal = dpt_normal_model(image)[0]
        normal = (normal * 255).astype(np.uint8).transpose(1, 2, 0)
        normal[~mask] = 0
        del dpt_normal_model

        final_normal = normal

    cv2.imwrite("_rgba.jpg", final_rgba)
    cv2.imwrite("_depth.jpg", final_depth)
    cv2.imwrite("_normal.jpg", final_normal)

    hist = cv2.calcHist([final_depth], [0], None, [256], [0, 256])
    # save histogram
    # plt.plot(range(1, 256), hist[1:])
    # plt.savefig("_hist.jpg")

    depth_peak_pixel = np.argmax(hist[1:])
    point_candis = np.where(final_depth == depth_peak_pixel)
    point_res = np.array(np.column_stack((point_candis[1], point_candis[0])))  # H, W -> W, H
    random_size = min(point_res.shape[0], 1)
    point_idx = np.random.choice(point_res.shape[0], size=random_size, replace=False)
    input_point = point_res[point_idx]



    # coords = np.nonzero(mask)
    # x_min, x_max = coords[0].min(), coords[0].max()
    # y_min, y_max = coords[1].min(), coords[1].max()

    # temp = np.zeros_like(final_rgba)
    # temp[x_min:x_max, y_min:y_max] = 255

    # cv2.imwrite("_temp.jpg", temp)



    # recenter
    # if opt.recenter:
    #     print(f'[INFO] recenter...')
    #     final_rgba = np.zeros((opt.size, opt.size, 4), dtype=np.uint8)
    #     final_depth = np.zeros((opt.size, opt.size), dtype=np.uint8)
    #     final_normal = np.zeros((opt.size, opt.size, 3), dtype=np.uint8)

    #     coords = np.nonzero(mask)
    #     x_min, x_max = coords[0].min(), coords[0].max()
    #     y_min, y_max = coords[1].min(), coords[1].max()
    #     h = x_max - x_min
    #     w = y_max - y_min
    #     desired_size = int(opt.size * (1 - opt.border_ratio))
    #     scale = desired_size / max(h, w)
    #     h2 = int(h * scale)
    #     w2 = int(w * scale)
    #     x2_min = (opt.size - h2) // 2
    #     x2_max = x2_min + h2
    #     y2_min = (opt.size - w2) // 2
    #     y2_max = y2_min + w2
    #     final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
    #     final_depth[x2_min:x2_max, y2_min:y2_max] = cv2.resize(depth[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
    #     final_normal[x2_min:x2_max, y2_min:y2_max] = cv2.resize(normal[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
    # else:
    #     final_rgba = carved_image
    #     final_depth = depth
    #     final_normal = normal


    sam_checkpoint = "./pretrained_sam_hq/sam_hq_vit_l.pth"
    model_type = "vit_l"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)

    hq_token_only = True  # multi object: False, single object: True
    predictor.set_image(image)


    # input_box = np.array([[y_min, x_min, y_max, x_max]])
    input_box = None
    
    input_label = np.ones(input_point.shape[0])

    result_path = './img/hq_sam_result/'
    os.makedirs(result_path, exist_ok=True)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box = input_box,
        multimask_output=False,
        hq_token_only=hq_token_only, 
    )
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_mask(masks[0], plt.gca())

    # show_points(input_point, input_label, plt.gca())

    # plt.axis('off')
    # plt.savefig(result_path + f'_example6' +'_'+'.png',bbox_inches='tight',pad_inches=-0.1)
    # plt.close()
    

    show_res(masks, scores, input_point, input_label, input_box, result_path + f'_example7', image)

    new_masks = (masks * 255).astype(np.uint8)
    new_masks = new_masks[0]
    binary = cv2.threshold(new_masks, 0, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    
    # ret, labels = cv2.connectedComponents(binary)
    # for label in range(1, ret):
    #     mask = np.array(labels, dtype=np.uint8)
    #     mask[labels == label] = 255
    #     cv2.imwrite(f"_temp{label}.jpg", mask)

    output = cv2.connectedComponentsWithStats(binary, 4, cv2.CV_32S)
    num_labels, labels, stats, centroids = output

    stats_idx = np.argmax(stats[..., -1][1:]) + 1  # remove background
    new_mask = np.array(labels, dtype=np.uint8)
    new_mask[labels == stats_idx] = 255

    # FIXME:
    cv2.imwrite("_original.jpg", masks[0] * 255)

    return new_mask  # foreground: white (255), background: black (0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SEGMENT')
    parser.add_argument("--path", type=str, default="./img/chair_0129.jpg")
    args = parser.parse_args()
    new_mask = segment_module(args.path)
    cv2.imwrite("_connected.jpg", new_mask)