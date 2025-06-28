#!/bin/bash
cd /home/hsb/prj_my/rs_mm

model=CoDF_ab
dataset=MUUFL

python train.py \
    dataset=$dataset \
    model=$model \
    dataset.img_size=[32,32] \
    model.training_settings.device=cuda:0 \
    model.training_settings.epochs=100 \
    model.training_settings.early_stop=500 \
    model.training_settings.train_batch_size=32 \
    model.training_settings.test_batch_size=32 \
    model.training_settings.num_workers=16 \
    model.training_settings.dropout=0.1\
    model.optimizer.lr=1e-2 \
    model.optimizer.weight_decay=5e-2 \
    model.training_settings.checkpoint=/home/data/hsb/checkpoint/${model}_${dataset}.pt

# epoch: 99  ,  train_loss: 0.0066  ,  test_loss: 0.1973   //  OA: 96.01 %  ,  AA: 91.80 %  ,  k: 94.72 %

#ablation:
# epoch: 99  ,  train_loss: 0.0012  ,  test_loss: 0.7079   //  OA: 89.35 %  ,  AA: 81.46 %  ,  k: 85.86 % 
#epoch: 99  ,  train_loss: 0.0068  ,  test_loss: 0.2004   //  OA: 95.80 %  ,  AA: 91.39 %  ,  k: 94.43 %