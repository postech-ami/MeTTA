#!/bin/bash

# single image
CUDA_VISIBLE_DEVICES=0 python utils/segment_grounded.py \
    /node_data/ugkim/sitto/src/rebuttal/misc/image/teddy_img.png \
    --single_image

# multi image (all)
# CUDA_VISIBLE_DEVICES=0 python utils/segment_grounded.py \
#     /node_data/ugkim/sitto/src/ohouse \
#     --border_ratio 0.25

# single directory
# CUDA_VISIBLE_DEVICES=0 python utils/segment_grounded.py \
#     /node_data/ugkim/sitto/src/rebuttal/sofa --border_ratio 0.25 \
#     --single_dir

