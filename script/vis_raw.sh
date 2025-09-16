#!/bin/bash
HYDRA_FULL_ERROR=1
cd /home/hsb/prj_my/LULC

model=CoDF
dataset=Houston2013 #Augsburg, MUUFL

python vis_raw.py \
    dataset=$dataset \
    model=$model \
    dataset.img_size=[32,32] \
    model.train.device=cuda:0 \
    model.train.save_dir=/home/data/hsb/checkpoint/LULC/

