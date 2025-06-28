import os
import cv2
import numpy as np
import torch
import matplotlib.patheffects as PathEffects
import tqdm
from torchvision import transforms
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import *

from train_co_meta import *
from sklearn.manifold import TSNE

def plot_pixel_tsne(data_loader, model, num_samples=10, perplexity=30):
    """
    参数说明：
    data_loader: 数据加载器
    model: 训练好的模型
    num_samples: 用于t-SNE的像素采样数
    perplexity: t-SNE的困惑度参数
    """
    model.eval()
    features_list = []
    labels_list = []
    device = next(model.parameters()).device  # 自动获取模型所在设备
    
    # 1. 从data_loader中提取特征和标签
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            # 假设data_loader返回图像和标签
            hsi, lidar, label, mask = batch
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            pred = model(hsi, lidar)
            features = torch.cat([model.p1[2],model.p2[2]],dim=1) # [B, C, H, W]

            # 获取批量大小和特征维度
            B, C, H, W = features.shape

            pred = pred.view(pred.shape[0], -1, H * W)  # 注意num_classes位置调整
            features = features.view(B, C, -1)
            label = label.view(label.shape[0], -1)
            mask = mask.view(mask.shape[0], -1)

            idx = torch.where(mask > 0)
            pred = pred[idx[0], :, idx[1]]
            label = label[idx]
            features = features[idx[0], :, idx[1]]

            features = features.cpu().numpy()
            label = label.cpu().numpy()

            features_reshaped = features
            labels_reshaped = label.reshape(-1)

            # 3. 随机采样（如果数据量太大）
            if len(features_reshaped) > num_samples:
                indices = np.random.choice(len(features_reshaped), num_samples, replace=False)
                features_reshaped = features_reshaped[indices]
                labels_reshaped = labels_reshaped[indices]
            
            features_list.append(features_reshaped)
            labels_list.append(labels_reshaped)
    
    # 4. 合并所有批次的特征和标签
    all_features = np.concatenate(features_list, axis=0)  # [N, C]
    all_labels = np.concatenate(labels_list, axis=0)      # [N]
    
    # 5. 运行t-SNE降维
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(all_features)
    
    # 6. 可视化
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图，不带热力图
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                         c=all_labels, cmap='tab20', s=5, alpha=0.5)
    
    # 计算每类点的中心并标注数字
    unique_labels = np.unique(all_labels)  # 获取所有类别（0-14）
    for label in unique_labels:
        # 提取属于该类别的点
        idx = (all_labels == label)
        class_points = tsne_results[idx]
        
        # 计算中心点（均值）
        center_x = np.mean(class_points[:, 0])
        center_y = np.mean(class_points[:, 1])
        
        # 在中心标注类别数字
        plt.text(center_x, center_y, str(int(label)), 
                fontsize=12, fontweight='bold', 
                ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # 去掉横纵坐标刻度
    plt.xticks([])  # 移除x轴刻度
    plt.yticks([])  # 移除y轴刻度
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('/home/wcy/temp_hsb/Co_Meta_seg/result/tsne/Houston2013/ba0.png', dpi=2000)
    plt.show()

if __name__ == '__main__':
    model.load_state_dict(torch.load('/home/wcy/temp_hsb/pth/co_meta_H_bad.pt',weights_only=True),False)

   
    plot_pixel_tsne(test_loader, model, num_samples=100, perplexity=30)