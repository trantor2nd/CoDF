import glob
from torch.utils import data
import numpy as np
import torch
from scipy.io import loadmat
from torchvision import transforms
import hydra
from omegaconf import DictConfig
class LULC_dataset(data.Dataset):
    def __init__(self, config : DictConfig, phase='train'):
        assert phase == 'train' or phase == 'test' , "phase should be train or test"
        self.config = config.dataset
        path_in = self.config.root

        path = path_in + '/' +phase
        self.files = glob.glob(path + '/*.mat')

        self.resize_transform = transforms.Resize(tuple(self.config.img_size))

    def __getitem__(self, item):
        data = loadmat(self.files[item])
        hsi = data['hsi']
        lidar = data['lidar']
        label = data['label']

        #label = label+1
        mask = (label >= 0).astype(np.int8)
        label[label < 0] = 0
        
        if len(hsi.shape) == 2:
            hsi = hsi[..., None]
        if len(lidar.shape) == 2:
            lidar = lidar[..., None]

        # Resize the images to 224x224
        hsi = self.resize_image(hsi.transpose(2, 0, 1), self.resize_transform)
        lidar = self.resize_image(lidar.transpose(2, 0, 1), self.resize_transform)
        label = self.resize_image(label, self.resize_transform, is_label=True)

        hsi = torch.from_numpy(hsi).type(torch.FloatTensor)
        lidar = torch.from_numpy(lidar).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.LongTensor)

        return hsi, lidar, label , mask

    def __len__(self):
        return len(self.files)

    def resize_image(self, image, transform, is_label=False):
        if is_label:
            # For labels, we need to ensure they remain as integers after resizing
            image = image.astype(np.uint8)
        image = torch.from_numpy(image).unsqueeze(0)  # Add batch dimension
        resized_image = transform(image).squeeze(0).numpy()
        if is_label:
            resized_image = resized_image.astype(np.int64)
        return resized_image