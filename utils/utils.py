from typing import Any, List, Tuple, Union
import torch
import numpy as np
import random
from PIL import Image
import os

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

def seed_everything(random_seed=42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def save_img(img, save_name):
    img = img.clone().detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).round().astype('uint8')

    W, H, C = img.shape
    if C == 1:
        img = Image.fromarray(img[:, :, 0])
    else:
        img = Image.fromarray(img)
    img = img.resize((H // 2, W // 2))
    img.save(os.path.join(f"{save_name}.png"))