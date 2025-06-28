from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def raw_pic(cfg:DictConfig):
    red = 630#700
    green = 520#546.1
    blue = 460#435.8

    UB = cfg.dataset.UB
    LB = cfg.dataset.LB
    bands = cfg.dataset.hsi_channel

    rn = int((red - LB)*bands / (UB-LB))
    gn = int((green - LB)*bands / (UB-LB)) 
    bn = int((blue - LB)*bands / (UB-LB)) 
    # 读取.mat文件
    hsi_data = sio.loadmat(cfg.dataset.data_begin.HSI.root)
    lidar_raw_data = sio.loadmat(cfg.dataset.data_begin.LiDAR.root)
  
    # 获取三个键的数据
    hsi_data = hsi_data[cfg.dataset.data_begin.HSI.key]
    lidar_raw_data = lidar_raw_data[cfg.dataset.data_begin.LiDAR.key]


    img = hsi_data[:,:,[rn,gn,bn]]
    # Convert hyperspectral data range to rgb data range
    img[:,:,0] = img[:,:,0]/np.max(img[:,:,0])*255
    img[:,:,1] = img[:,:,1]/np.max(img[:,:,1])*255
    img[:,:,2] = img[:,:,2]/np.max(img[:,:,2])*255
    hsi_img = np.ceil(img)
    
    if len(lidar_raw_data.shape) == 3 :
        lidar_data = lidar_raw_data.mean(-1)
    else :
        lidar_data = lidar_raw_data
    
    low, high = np.percentile(lidar_data, [1, 99])
    lidar_data = np.clip(lidar_data, low, high)

    lidar_data = (lidar_data - low) / (high - low + 1e-8)
    lidar_data = lidar_data * 255

    if cfg.dataset.name == 'Augsburg':
        sar_data =  lidar_raw_data
        data = sar_data-sar_data.min()  # 将负值截断为 0，SAR 数据通常非负
        data = np.log1p(data)  # log(1 + x) 避免 log(0)
        # 归一化到 [0, 1] 范围
        processed_data = np.zeros_like(data, dtype=np.float32)
        for ch in range(4):
            ch_data = data[:,:,ch]
            ch_min = np.nanmin(ch_data)
            ch_max = np.nanmax(ch_data)
            if ch_max > ch_min:
                processed_data[:,:,ch] = (ch_data - ch_min) / (ch_max - ch_min)
            else:
                processed_data[:,:,ch] = np.zeros_like(ch_data)

        rgb_data = processed_data.mean(axis=-1)
        # 转换为 (H, W, 3) 格式并缩放到 [0, 255]

        rgb_data = np.nan_to_num(rgb_data, nan=0.0) * 255
        lidar_data = np.clip(rgb_data, 0, 255)

    out_path = './result/' + cfg.dataset.name 
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Created directory: {out_path}")
    else:
        print(f"Directory already exists: {out_path}")

    # 可视化并保存为PNG文件
    Image.fromarray(hsi_img.astype(np.uint8)).save(out_path +'/HSI.png')
    Image.fromarray(lidar_data.astype(np.uint8)).save(out_path +'/LiDAR.png')

  


if __name__ == '__main__':
    raw_pic()
