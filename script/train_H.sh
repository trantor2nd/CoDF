#!/bin/bash
cd /home/hsb/prj_my/rs_mm

model=CoDF_ab
dataset=Houston2013

python train.py \
    dataset=$dataset \
    model=$model \
    dataset.img_size=[32,32] \
    model.training_settings.device=cuda:2 \
    model.training_settings.epochs=100 \
    model.training_settings.early_stop=500 \
    model.training_settings.train_batch_size=16 \
    model.training_settings.test_batch_size=16 \
    model.training_settings.num_workers=16 \
    model.training_settings.dropout=0.1\
    model.optimizer.lr=1e-4 \
    model.optimizer.weight_decay=1e-2 \
    model.training_settings.checkpoint=/home/data/hsb/checkpoint/${model}_${dataset}.pt
#epoch: 99  ,  train_loss: 0.0249  ,  test_loss: 0.1333   //  OA: 98.04 %  ,  AA: 98.06 %  ,  k: 97.90 %
#ablation:
#epoch: 99  ,  train_loss: 0.0910  ,  test_loss: 0.3679   //  OA: 89.96 %  ,  AA: 89.98 %  ,  k: 89.25 % 
#epoch: 99  ,  train_loss: 0.0245  ,  test_loss: 0.1683   //  OA: 97.71 %  ,  AA: 97.73 %  ,  k: 97.54 %
#epoch: 99  ,  train_loss: 0.1238  ,  test_loss: 0.2901   //  OA: 90.87 %  ,  AA: 90.85 %  ,  k: 90.22 %