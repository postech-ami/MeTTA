import subprocess
import shlex
import os
from glob import glob

# classes = ["bed", "bookcase", "desk", "misc", "sofa", "table", "chair"]
classes = ["chair"]
for cls in classes:
    img_pathes = glob(f"/data/sitto/real/{cls}/image/real*.jpg")
    # img_pathes += glob(f"/data/sitto/pix3d/{cls}/image/pix3d_*.png")
    for img_path in img_pathes:
        img_path = f"shells/preprocess_image.sh '{img_path}'"
        subprocess.call(shlex.split(img_path))        

# subprocess.call(shlex.split("shells/preprocess_image.sh './img/cactus.png'"))
