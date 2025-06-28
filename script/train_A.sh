#!/bin/bash
cd /home/hsb/prj_my/rs_mm

model=CoDF_ab
dataset=Augsburg

python train.py \
    dataset=$dataset \
    model=$model \
    dataset.img_size=[32,32] \
    model.training_settings.device=cuda:0 \
    model.training_settings.epochs=200 \
    model.training_settings.early_stop=500 \
    model.training_settings.train_batch_size=32 \
    model.training_settings.test_batch_size=32 \
    model.training_settings.num_workers=16 \
    model.training_settings.dropout=0.1\
    model.optimizer.lr=1e-3 \
    model.optimizer.weight_decay=1e-2 \
    model.training_settings.checkpoint=/home/data/hsb/checkpoint/${model}_${dataset}.pt

#epoch: 199  ,  train_loss: 0.0122  ,  test_loss: 0.0843   //  OA: 97.74 %  ,  AA: 90.66 %  ,  k: 96.74 % 

#epoch: 199  ,  train_loss: 0.1659  ,  test_loss: 0.2198   //  OA: 92.83 %  ,  AA: 68.38 %  ,  k: 89.52 %
#epoch: 199  ,  train_loss: 0.0208  ,  test_loss: 0.0904   //  OA: 97.58 %  ,  AA: 89.40 %  ,  k: 96.52 %