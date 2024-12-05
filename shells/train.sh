#!/bin/bash

wandb enabled

_CLASS="chair"
_ID="0178"
_DATA="pix3d_im3d"
_geo_lr=0.001
_tex_lr=0.001
_loc_lr=0.0001
_sdf=0.6
_normal=2
_lap=2
_wandb_user="ug-kim"

## 1-stage
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config configs/main.json \
    --out_dir ${_DATA}_${_CLASS}_id-${_ID} \
    --save_interval 100 \
    --sds_interval 100 \
    --geo_lr ${_geo_lr} \
    --tex_lr ${_tex_lr} \
    --loc_lr ${_loc_lr} \
    --geo_range 0 0.3 \
    --tex_range 0 0.3 \
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