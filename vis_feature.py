import numpy as np
import torch
import tqdm
from scipy.io import loadmat
import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from  torchvision import utils as vutils
from PIL import Image
import os

import model

colormap = np.array(
    [
        [0, 0, 0],
        [0, 0, 205],
        [0, 8, 255],
        [0, 77, 255],
        [0, 145, 255],
        [0, 212, 255],
        [41, 255, 206],
        [96, 255, 151],
        [151, 255, 96],
        [206, 255, 41],
        [255, 230, 0],
        [255, 167, 0],
        [255, 104, 0],
        [255, 41, 0],
        [205, 0, 0],
        [128, 0, 0]
    ]
)


def nor(img):
    img -= np.min(img)
    img = img / np.max(img)
    return img

def nor_resize(img, size) :
    #img (H,W) or (H,W,C)
    if len(img.shape) == 2:
        img = np.pad(nor(img), (size // 2, size // 2), constant_values=0)[..., None] #-> (H,W,1)
    else:
        img = np.stack([np.pad(nor(img[..., i]), (size // 2, size // 2), constant_values=0)
                         for i in range(img.shape[-1])], axis=-1)
    return img

def get_inputs_data(cfg: DictConfig):
    hsi = loadmat(cfg.dataset.data_begin.HSI.root)[cfg.dataset.data_begin.HSI.key]
    lidar = loadmat(cfg.dataset.data_begin.LiDAR.root)[cfg.dataset.data_begin.LiDAR.key]

    hsi = nor_resize(hsi, cfg.dataset.img_size[0])    
    lidar = nor_resize(lidar, cfg.dataset.img_size[0])

    print(hsi.shape,lidar.shape)
    return hsi, lidar

def crop_img2model(
    model,
    device,
    hsi,
    lidar,
    crop_size
) :
    pad_x = crop_size - hsi.shape[0] % crop_size
    pad_y = crop_size - hsi.shape[1] % crop_size

    x = hsi.shape[0] + pad_x % crop_size
    y = hsi.shape[1] + pad_y % crop_size

    hsi_pad = np.zeros(shape=(x, y, hsi.shape[-1] if len(hsi.shape) > 2 else 1))
    hsi_pad[pad_x % crop_size:, pad_y % crop_size:] = hsi
    lidar_pad = np.zeros(shape=(x, y, lidar.shape[-1] if len(lidar.shape) > 2 else 1))
    lidar_pad[pad_x % crop_size:, pad_y % crop_size:] = lidar

    pics = 7 # 7 features
    outs = torch.zeros(pics,x,y)
    for i in tqdm.tqdm(range(crop_size // 2, x - crop_size // 2 + 1, crop_size)):
        for j in range(crop_size // 2, y - crop_size // 2 + 1, crop_size):
            crop_hsi = hsi_pad[i - crop_size // 2: i + crop_size // 2, j - crop_size // 2: j + crop_size // 2]
            crop_lidar = lidar_pad[i - crop_size // 2: i + crop_size // 2, j - crop_size // 2: j + crop_size // 2]

            crop_hsi = torch.from_numpy(crop_hsi.transpose(2, 0, 1)).type(torch.FloatTensor)
            crop_lidar = torch.from_numpy(crop_lidar.transpose(2, 0, 1)).type(torch.FloatTensor)

            with torch.no_grad():
                crop_hsi = crop_hsi.to(device)
                crop_lidar = crop_lidar.to(device)
                _ = model(crop_hsi[None], crop_lidar[None])
        
            #--------------------------------------------------------
            f=[model.p1[0],model.p2[0],model.p1[1],model.p2[1],model.p1[2],model.p2[2],model.p3]

            for _ in range(pics) :
                probe = f[_]
                probe = probe.mean(dim=1)[0].cpu()
                outs[_, i - crop_size // 2: i + crop_size // 2, j - crop_size // 2: j + crop_size // 2] = probe            
            #--------------------------------------------------------   
    outs = outs[:,pad_x % crop_size:, pad_y % crop_size:]
    pic_out = outs[:, crop_size // 2: -crop_size // 2, crop_size // 2: -crop_size // 2]


    return pic_out , pics
    
    

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    #-----------------------------------------------------------------
    model_path = cfg.model.train.save_dir + cfg.model.name +'/'+ cfg.dataset.name + '/best_model.pt'
    crop_size = cfg.dataset.img_size[0]
    #-----------------------------------------------------------------

    #-----------------------------------------------------------------
    device = torch.device(cfg.model.train.device if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())

    model = hydra.utils.instantiate(
        cfg.model.use_model
    ).to(device)

    model.load_state_dict(torch.load(model_path,weights_only=True),False)
    model.eval()
    #-----------------------------------------------------------------

    hsi, lidar = get_inputs_data(cfg)

    pic_out,pics = crop_img2model(model, device, hsi, lidar, crop_size)

    out_path = './result/' + cfg.dataset.name + '/' + cfg.model.name
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f"Created directory: {out_path}")
    else:
        print(f"Directory already exists: {out_path}")

    for _ in range(pics) :
        vutils.save_image(pic_out[_], out_path+'/'+ f'feature{_}' +'.png',normalize=True)

if __name__ == '__main__':
   main()

    
