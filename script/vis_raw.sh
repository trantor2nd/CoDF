#!/bin/bash
cd /home/hsb/prj_my/rs_mm

model=CoDF
dataset=Houston2013

python vis_raw.py \
    dataset=$dataset \
    model=$model \
    dataset.img_size=[32,32] \
    model.training_settings.device=cuda:0 \
    model.training_settings.dropout=0.1\
    model.training_settings.checkpoint=/home/data/hsb/checkpoint/${model}_${dataset}.pt

