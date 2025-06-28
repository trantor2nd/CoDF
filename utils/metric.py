import torch
import math
import os
import random
import numpy as np

def compute_confusion_matrix(pred, label, num_classes): 
    with torch.no_grad():
        pred = pred.long().view(-1)
        label = label.long().view(-1)

        # 确保张量在同一设备
        device = pred.device
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

        # 计算 indices 并展平
        indices = (label * num_classes + pred).view(-1)

        # 确保 ones_like 的张量在正确设备
        confusion_matrix = confusion_matrix.view(-1).scatter_add_(
            0, indices, torch.ones_like(indices, device=device)
        ).view(num_classes, num_classes)

        return confusion_matrix


def calculate_metrics(confusion_matrix):
    # 将混淆矩阵转换为浮点类型
    confusion_matrix = confusion_matrix.float()
    
    # 计算对角线上的值之和（正确预测的数量）
    oa = confusion_matrix.trace() / confusion_matrix.sum()
    
    # 计算平均精度（每个类别的精度的平均值）
    precision_per_class = confusion_matrix.diag() / (confusion_matrix.sum(1)+1e-3)

    aa = precision_per_class.mean()#去掉背景
    #print("per_class_accuracy:")
    #print(precision_per_class)
    # 计算Kappa系数

    total_sum = confusion_matrix.sum()
    actual_sum = confusion_matrix.sum(1)
    pred_sum = confusion_matrix.sum(0)
    
    expected_sum = torch.dot(actual_sum, pred_sum) / total_sum
    kappa = ( confusion_matrix.trace() / confusion_matrix.sum() - expected_sum/total_sum) / (1 - expected_sum/total_sum)
    
    return oa.item(), aa.item(), kappa.item()
