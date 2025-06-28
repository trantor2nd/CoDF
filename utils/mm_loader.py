import glob
from torch.utils import data
import numpy as np
import torch
from scipy.io import loadmat
from torchvision import transforms
import hydra
from omegaconf import DictConfig
class mm_dataset(data.Dataset):
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




if __name__ == '__main__':

    def count_classes(dataset, num_classes):
        class_counts = [0] * num_classes
        for i in range(len(dataset)):
            _,_,label,_ = dataset[i]
            unique, counts = np.unique(label.numpy(), return_counts=True)
            for u, c in zip(unique, counts):
                class_counts[u] += c
        return class_counts



    x = mm_dataset(name='Houston2013')
    print(x[0][0].shape,x[0][1].shape,len(x),x[0][2].shape)
    x = mm_dataset(name='Augsburg')
    print(x[0][0].shape,x[0][1].shape,len(x),x[0][2].shape)
    x = mm_dataset(name='MUUFL')
    print(x[0][0].shape,x[0][1].shape,len(x),x[0][2].shape)

    for data_name, num_classes in [('Augsburg',7) ,('MUUFL',11)]:#('Houston2013',15) ,
        
        train_dataset = mm_dataset(name=data_name, phase='train')
        test_dataset = mm_dataset(name=data_name, phase='test')

        # 统计训练集和测试集的类别总数
        train_class_counts = count_classes(train_dataset, num_classes+1)
        test_class_counts = count_classes(test_dataset, num_classes+1)

        print(f"Train set class counts: {train_class_counts}")
        print(f"Test set class counts: {test_class_counts}")
    