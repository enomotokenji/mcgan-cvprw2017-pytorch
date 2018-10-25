import glob
import cv2
import random
import numpy as np
import pickle
import os

from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        with open(config.imlist, 'rb') as f:
            self.imlist = pickle.load(f, encoding='latin-1')

        cloud_files = glob.glob(os.path.join(config.cloud_dir, '*.png'))
        self.cloud_files = cloud_files
        self.n_cloud = len(cloud_files)

    def __getitem__(self, index):
        rgb = cv2.imread(os.path.join(self.config.rgbnir_dir, 'RGB', str(self.imlist[index])), 1).astype(np.float32)
        nir = cv2.imread(os.path.join(self.config.rgbnir_dir, 'NIR', str(self.imlist[index])), 0).astype(np.float32)
        cloud = cv2.imread(self.cloud_files[random.randrange(self.n_cloud)], -1).astype(np.float32)

        alpha = cloud[:, :, 3] / 255.
        alpha = np.broadcast_to(alpha[:, :, None], alpha.shape + (3,))
        cloud_rgb = (1. - alpha) * rgb + alpha * cloud[:, :, :3]
        cloud_rgb = np.clip(cloud_rgb, 0., 255.)

        cloud_mask = cloud[:, :, 3]

        x = np.concatenate((cloud_rgb, nir[:, :, None]), axis=2)
        t = np.concatenate((rgb, cloud_mask[:, :, None]), axis=2)

        x = x / 127.5 - 1
        t = t / 127.5 - 1

        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)

        return x, t

    def __len__(self):

        return len(self.imlist)

class TestDataset(data.Dataset):
    def __init__(self, test_dir):
        super().__init__()
        self.test_dir = test_dir
        
        self.test_files = glob.glob(os.path.join(test_dir, 'RGB', '*.png'))

    def __getitem__(self, index):
        filename = os.path.basename(self.test_files[index])
        cloud_rgb = cv2.imread(os.path.join(self.test_dir, 'RGB', filename), 1).astype(np.float32)
        nir = cv2.imread(os.path.join(self.test_dir, 'NIR', filename), 0).astype(np.float32)

        x = np.concatenate((cloud_rgb, nir[:, :, None]), axis=2)

        x = x / 127.5 - 1

        x = x.transpose(2, 0, 1)

        return x

    def __len__(self):

        return len(self.test_files)
