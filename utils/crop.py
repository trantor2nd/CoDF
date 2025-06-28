import os
import numpy as np
from scipy.io import loadmat, savemat
import tqdm

def nor(img):
    img -= np.min(img)
    img = img / np.max(img)
    return img


def crop_data(img1, img2, Trlabel,Tslabel, path_out,crop_size):

    if len(img1.shape) == 2:
        img1 = np.pad(nor(img1), (crop_size // 2, crop_size // 2), constant_values=0)
    else:
        img1 = np.stack([np.pad(nor(img1[..., i]), (crop_size // 2, crop_size // 2), constant_values=0)
                         for i in range(img1.shape[-1])], axis=-1)
    if len(img2.shape) == 2:
        img2 = np.pad(nor(img2), (crop_size // 2, crop_size // 2), constant_values=0)
    else:
        img2 = np.stack([np.pad(nor(img2[..., i]), (crop_size // 2, crop_size // 2), constant_values=0)
                         for i in range(img2.shape[-1])], axis=-1)
   
    x, y = Trlabel.shape
    Trlabel = np.pad(Trlabel, (crop_size // 2, crop_size // 2), constant_values=0)
    Tslabel = np.pad(Tslabel, (crop_size // 2, crop_size // 2), constant_values=0)

    ii = 0
    #train dataset
    for i in tqdm.tqdm(range(crop_size // 2, x + crop_size // 2)):
        for j in range(crop_size // 2, y + crop_size // 2):
            if Trlabel[i,j] == 0:
                continue
            path = path_out+ '/train'
            os.makedirs(path, exist_ok=True)
            savemat(path + '/' + str(ii) + '.mat', {
                'hsi': img1[i - crop_size // 2: i + crop_size // 2+crop_size%2, j - crop_size // 2: j + crop_size // 2+crop_size%2],
                'lidar': img2[i - crop_size // 2: i + crop_size // 2+crop_size%2, j - crop_size // 2: j + crop_size // 2+crop_size%2],
                'label': Trlabel[i,j].astype(np.int8) - 1, # first class : 0
            })
            ii += 1

    ii = 0
    #test dataset
    for i in tqdm.tqdm(range(crop_size // 2, x + crop_size // 2)):
        for j in range(crop_size // 2, y + crop_size // 2):
            if Tslabel[i,j] == 0:
                continue
            path = path_out+ '/test'
            os.makedirs(path, exist_ok=True)
            savemat(path + '/' + str(ii) + '.mat', {
                'hsi': img1[i - crop_size // 2: i + crop_size // 2+crop_size%2, j - crop_size // 2: j + crop_size // 2+crop_size%2],
                'lidar': img2[i - crop_size // 2: i + crop_size // 2+crop_size%2, j - crop_size // 2: j + crop_size // 2+crop_size%2],
                'label':  Tslabel[i,j].astype(np.int8) - 1, # first class : 0
            })
            ii += 1


def crop_Houston2013(crop_size):
    hsi = loadmat('/home/data/shared_datasets/Houston2013/raw/HSI.mat')['HSI']
    lidar = loadmat('/home/data/shared_datasets/Houston2013/raw/LiDAR_1.mat')['LiDAR']
    train_label = loadmat('/home/data/shared_datasets/Houston2013/raw/TRLabel.mat')['TRLabel']
    test_label = loadmat('/home/data/shared_datasets/Houston2013/raw/TSLabel.mat')['TSLabel']

    print(train_label.min(), train_label.max())
    print(test_label.min(), test_label.max())

    path_out = '/home/data/shared_datasets/Houston2013/'+'{}x{}'.format(crop_size,crop_size)
    os.makedirs(path_out, exist_ok=True)
    crop_data(hsi, lidar, train_label,test_label, path_out,crop_size )

def crop_MUUFL(crop_size):
    hsi = loadmat('/home/data/shared_datasets/MUUFL/raw/hsi.mat')['HSI']
    lidar = loadmat('/home/data/shared_datasets/MUUFL/raw/lidar.mat')['lidar']
    labels = loadmat('/home/data/shared_datasets/MUUFL/raw/train_test_gt.mat')
    train_label = labels['trainlabels']
    test_label = labels['testlabels']

    print(train_label.min(), train_label.max())
    print(test_label.min(), test_label.max())

    path_out = f'/home/data/shared_datasets/MUUFL/' + '{}x{}'.format(crop_size,crop_size)
    os.makedirs(path_out, exist_ok=True)
    crop_data(hsi, lidar, train_label,test_label, path_out,crop_size )


def crop_Augsburg(crop_size):
    hsi = loadmat('/home/data/shared_datasets/Augsburg/raw/data_hsi.mat')['data']
    sar = loadmat('/home/data/shared_datasets/Augsburg/raw/data_sar.mat')['data']
    train_label = loadmat('/home/data/shared_datasets/Augsburg/raw/mask_train.mat')['mask_train']
    test_label = loadmat('/home/data/shared_datasets/Augsburg/raw/mask_test.mat')['mask_test'].astype(np.uint8)

    print(train_label.min(), train_label.max())
    print(test_label.min(), test_label.max())

    path_out = '/home/data/shared_datasets/Augsburg/' + '{}x{}'.format(crop_size,crop_size)
    os.makedirs(path_out, exist_ok=True)
    crop_data(hsi, sar, train_label,test_label, path_out,crop_size )

def crop_Trento(crop_size):
    hsi = loadmat('/home/data/shared_datasets/Trento/raw/HSI.mat')['HSI']
    lidar = loadmat('/home/data/shared_datasets/Trento/raw/LiDAR.mat')['LiDAR']
    train_label = loadmat('/home/data/shared_datasets/Trento/raw/TRLabel.mat')['TRLabel']
    test_label = loadmat('/home/data/shared_datasets/Trento/raw/TSLabel.mat')['TSLabel']

    print(train_label.min(), train_label.max())
    print(test_label.min(), test_label.max())

    path_out = f'/home/data/shared_datasets/Trento/' + '{}x{}'.format(crop_size,crop_size)
    os.makedirs(path_out, exist_ok=True)
    crop_data(hsi, lidar, train_label,test_label, path_out,crop_size )

if __name__ == '__main__':
    # crop_Houston2013(7)
    # crop_MUUFL(7)
    # crop_Augsburg(7)
    crop_Trento(7)