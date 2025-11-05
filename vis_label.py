import os
import numpy as np
from scipy.io import loadmat
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf


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
        [128, 0, 0],
    ]
)


def load_labels_for_dataset(cfg: DictConfig):
    name = cfg.dataset.name
    # Default placeholders
    train = None
    test = None

    if name == 'Houston2013':
        train = loadmat(cfg.dataset.data_begin.train_label.root)[cfg.dataset.data_begin.train_label.key]
        test = loadmat(cfg.dataset.data_begin.test_label.root)[cfg.dataset.data_begin.test_label.key]

    elif name.upper() in ('MUUFL', 'MUUFL'):
        # file contains keys 'trainlabels' and 'testlabels'
        labels = loadmat('/home/data/shared_datasets/MUUFL/raw/train_test_gt.mat')
        # support both lowercase and lowercase variants
        if 'trainlabels' in labels:
            train = labels['trainlabels']
        elif 'trainlabels' in labels:
            train = labels['trainlabels']
        if 'testlabels' in labels:
            test = labels['testlabels']
        elif 'testlabels' in labels:
            test = labels['testlabels']

    elif name == 'Augsburg' or name.lower() == 'augsburg':
        train = loadmat('/home/data/shared_datasets/Augsburg/raw/mask_train.mat')['mask_train']
        test = loadmat('/home/data/shared_datasets/Augsburg/raw/mask_test.mat')['mask_test'].astype(np.uint8)

    elif name == 'Trento' or name.lower() == 'trento':
        # follow the same naming convention used elsewhere if available in cfg
        try:
            train = loadmat(cfg.dataset.data_begin.train_label.root)[cfg.dataset.data_begin.train_label.key]
            test = loadmat(cfg.dataset.data_begin.test_label.root)[cfg.dataset.data_begin.test_label.key]
        except Exception:
            # fallback to common Trento naming used in scripts
            train = loadmat('/home/data/shared_datasets/Trento/raw/TRLabel.mat')['TRLabel']
            test = loadmat('/home/data/shared_datasets/Trento/raw/TSLabel.mat')['TSLabel']

    else:
        # Attempt to read generic keys from config if present
        try:
            train = loadmat(cfg.dataset.data_begin.train_label.root)[cfg.dataset.data_begin.train_label.key]
            test = loadmat(cfg.dataset.data_begin.test_label.root)[cfg.dataset.data_begin.test_label.key]
        except Exception as e:
            raise RuntimeError(f"Unsupported dataset '{name}' and failed to locate label files: {e}")

    return train, test


def merge_train_test(train_label, test_label):
    # Ensure both labels are numpy arrays and same shape
    if train_label is None or test_label is None:
        raise ValueError('Both train_label and test_label are required')

    # If shapes differ, try squeezing or transposing common mistakes
    if train_label.shape != test_label.shape:
        # try transpose
        if train_label.T.shape == test_label.shape:
            train_label = train_label.T
        elif test_label.T.shape == train_label.shape:
            test_label = test_label.T
        else:
            raise ValueError(f"Train/test label shapes mismatch: {train_label.shape} vs {test_label.shape}")

    # Combined: prefer test label where available, otherwise train label
    combined = np.array(train_label, copy=True)
    mask = (test_label != 0)
    combined[mask] = test_label[mask]
    return combined


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_label, test_label = load_labels_for_dataset(cfg)

    print('train_label range:', np.min(train_label), np.max(train_label))
    print('test_label range:', np.min(test_label), np.max(test_label))

    combined = merge_train_test(train_label, test_label)

    # Map combined labels to colormap. We assume labels are small positive integers; 0 is background.
    max_label = int(combined.max())
    if max_label >= len(colormap):
        # If there are more classes than our colormap rows, tile the colormap
        reps = (max_label // len(colormap)) + 1
        cmap = np.tile(colormap, (reps, 1))
    else:
        cmap = colormap

    rgb = cmap[combined.astype(np.int64)]

    out_path = './result/' + cfg.dataset.name
    os.makedirs(out_path, exist_ok=True)

    Image.fromarray(rgb.astype(np.uint8)).save(out_path + '/label.png')
    print('Saved combined label to', out_path + '/label.png')


if __name__ == '__main__':
    main()
