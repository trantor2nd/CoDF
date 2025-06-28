#!/bin/bash
cd /home/hsb/prj_my/rs_mm

model=CoDF
dataset=Trento

python train.py \
    dataset=$dataset \
    model=$model \
    dataset.img_size=[32,32] \
    model.training_settings.device=cuda:0 \
    model.training_settings.epochs=20 \
    model.training_settings.early_stop=50 \
    model.training_settings.train_batch_size=16 \
    model.training_settings.test_batch_size=16 \
    model.training_settings.num_workers=16 \
    model.training_settings.dropout=0.4\
    model.optimizer.lr=1e-5 \
    model.optimizer.weight_decay=1e-2 \
    model.training_settings.checkpoint=/home/data/hsb/checkpoint/${model}_${dataset}.pt

#epoch: 19  ,  train_loss: 0.4146  ,  test_loss: 0.3044   //  OA: 97.13 %  ,  AA: 94.84 %  ,  k: 96.15 % 