#!/bin/bash
cd /home/hsb/prj_my/LULC

model=CoDF
dataset=Houston2013

python train.py \
    dataset=$dataset \
    model=$model \
    dataset.img_size=[32,32] \
    model.train.device=cuda:2 \
    model.train.epochs=100 \
    model.train.batch_size=16 \
    model.train.num_workers=16 \
    model.train.dropout=0.1\
    model.optimizer.lr=1e-4 \
    model.optimizer.weight_decay=1e-2 \
    model.train.save_dir=/home/data/hsb/checkpoint/LULC/