# MeTTA: Single-View to 3D Textured Mesh Reconstruction with Test-Time Adaptation (BMVC 2024)

### [Paper](https://arxiv.org/abs/2408.11465) | [Project Page](https://metta3d.github.io/)

### Install
- Python version 3.9.
```bash
# conda setup
conda create -n metta python=3.9
conda activate metta
```
- You can use cuda that fits your gpu environment.
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
```

- Install cuDNN followd by [this guideline](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

- Install additional pacakges.
```bash
pip install -r requirements.txt

# for zero123
pip install taming-transformers-rom1504 --no-deps
```

### Trouble-shooting
- Install gridencoder manually
```bash
pip install ./gridencoder
```

- If there is a problem with the ninja package, check if OpenGL is installed on your gpu. 
```bash
sudo apt-get install -y build-essential
sudo apt-get install freeglut3-dev libglu1-mesa-dev mesa-common-dev
```

### Prerequisite
To use multi-view diffusion models, you need to download some pretrained checkpoints.
- [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) for multi-view diffusion priors. We use `zero123-xl.ckpt`.

```bash
cd pretrained/zero123
wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
```

- [Omnidata](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch) for depth and normal prediction.

```bash
mkdir pretrained/omnidata
cd pretrained/omnidata
# assume gdown is installed
gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt
```

### Prerequisite (not mandatory)
Before the optimization, you need to get object-centric segmented images from real scene images.
We use [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to get segmentation images. You can also use your personal segmentation modules. 

- We use [SAM-HQ](https://github.com/SysCV/sam-hq) for segmentation models. Please download `sam_hq_vit_h.pth` in the below directory.

```bash
cd pretrained/sam
```

- Install [SAM-HQ](https://github.com/SysCV/sam-hq) followed by their github repository.
```bash
pip install segment-anything-hq
```


- Install Ground-Segmet-Anything followed by their github repository.
```bash
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git --recursive
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-11.8/  # your cuda version

pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install --upgrade diffusers[torch]
git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh
```

```bash
sh shells/preprocess_grounded_sam.sh
```

The shell file is organized as follows.
```bash
python utils/segment_grounded.py \
    {your_image_path} \
    --single_image
```


### Usage
First you need to create a config file under `configs`, you can copy and edit one of the provided examples like this:
```json
{
    "random_textures": true,
    "iter": 1500,
    "texture_res": [ 1024, 1024 ],
    "train_res": [512, 512],
    "batch": 8,
    "learn_light" : false,
    "envmap": "data/irrmaps/mud_road_puresky_4k.hdr",
    "pixel_loss": true,
    "write_video": true,
    "hard_time": true,
    "optim_radius": true,
    "train_location": true
}
```

The default settings are tested under a 48GB A6000. 
Lower `batch` and increase `iter` if your GPU memory is limited.

Then you can run training by:
```bash
# single GPU
sh shells/train.sh
```

The shell file is organized as follows:

```bash
# Your input image file path: "./src/pix3d_im3d/chair/image/pix3d_chair_0178_img_rgba.png"
# It can be converted to "./src/{_DATA}/{_CLASS}/image/pix3d_chair_{_ID}_img_rgba.png"

_CLASS="chair"  # category name
_ID="0178"  # ID in the input file name
_DATA="pix3d_im3d"  # folder name
_geo_lr=0.001
_tex_lr=0.001
_loc_lr=0.0001
_sdf=0.6
_normal=2
_lap=2
_wandb_user=temp # your user name

## 1-stage
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config configs/main.json \
    --out_dir ${_DATA}_${_CLASS}_id-${_ID} \
    --save_interval 100 \
    --sds_interval 100 \
    --geo_lr ${_geo_lr} \
    --tex_lr ${_tex_lr} \
    --loc_lr ${_loc_lr} \
    --geo_range 0 1 \
    --tex_range 0 1 \
    --geo_schedule 0.4 \
    --init_mesh_thicker 0 \
    --pix3d_class ${_CLASS} \
    --pix3d_id ${_ID} \
    --lambda_rgb 1 \
    --lambda_mask 1 \
    --data_type ${_DATA} \
    --radius 3.2 \
    --sdf_regularizer ${_sdf} \
    --lambda_mesh_normal ${_normal} \
    --lambda_mesh_laplacian ${_lap} \
    --wandb_user ${_wandb_user}
```

For single GPU, it takes about 30 minutes to train a single model (1500 iters at batch size of 8).

The validation/checkpoints/final mesh will be stored to `./work_dirs/<out_dir>`.



### Acknowledgement
* The awesome original paper:
```bibtex
@misc{chen2023fantasia3d,
      title={Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation}, 
      author={Rui Chen and Yongwei Chen and Ningxin Jiao and Kui Jia},
      year={2023},
      eprint={2303.13873},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{liu2023zero1to3,
      title={Zero-1-to-3: Zero-shot One Image to 3D Object}, 
      author={Ruoshi Liu and Rundi Wu and Basile Van Hoorick and Pavel Tokmakov and Sergey Zakharov and Carl Vondrick},
      year={2023},
      eprint={2303.11328},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

* [Fantasia3D unofficial](https://github.com/ashawkey/fantasia3d.unofficial) codebase.

* [Nvdiffrec](https://github.com/NVlabs/nvdiffrec) codebase.