import os
from glob import glob
import imageio
from PIL import Image

out_dir = "/node_data/ugkim/fantasia3d/work_dirs/0163_img_circle-chair_lr-0.001-0.01_optim-2_normal"
out_dir_name = out_dir.split("/")[-1]
vis_out_dir = os.path.join(out_dir, "front")
all_vis = glob(os.path.join(vis_out_dir, "*.png"))
all_vis.sort()

all_vis = [Image.open(vis) for vis in all_vis]

imageio.mimwrite(os.path.join(out_dir, "video", f"{out_dir_name}_front-view.mp4"), all_vis, fps=10, quality=8, macro_block_size=1)