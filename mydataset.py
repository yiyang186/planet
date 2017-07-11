import torch
import utils
from torch.utils import data
from PIL import Image
from scipy import misc
import numpy as np
from scipy import ndimage
from skimage.transform import warp, AffineTransform

class PlanetDataset(data.Dataset):
    
    def __init__(self, files=None, labels=None, test=False, transform=None, pic_size=(64, 64)):
        self.test = test  # training set or test set
        self.transform = transform
        self.pic_size = pic_size
        if self.test is False:
            self.train_files = files
            self.train_labels = labels
        else:
            self.test_files = files

    def __getitem__(self, index):
        if self.test is False:
            target = self.train_labels[index]
            img = Image.open('E:/new_data/kaggle/planet/train-jpg/{}.jpg'.format(self.train_files[index])).convert('RGB')
            img = self.transform(img, self.pic_size)
            return img, target
        else:
            img = Image.open('E:/new_data/kaggle/planet/test-jpg/{}.jpg'.format(self.test_files[index])).convert('RGB')
            # img = Image.open('E:/new_data/kaggle/planet/train-jpg/{}.jpg'.format(self.test_files[index])).convert('RGB')
            img = self.transform(img, self.pic_size)
            return img

    def __len__(self):
        if self.test is False:
            return self.train_files.shape[0]
        else:
            return self.test_files.shape[0]

class RandomAffineTransform(object):
    def __init__(self,
                 scale_range,
                 rotation_range,
                 shear_range,
                 translation_range
                 ):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.translation_range = translation_range

    def __call__(self, img):
        img_data = np.array(img)
        h, w, n_chan = img_data.shape
        scale_x = np.random.uniform(*self.scale_range)
        scale_y = np.random.uniform(*self.scale_range)
        scale = (scale_x, scale_y)
        rotation = np.random.uniform(*self.rotation_range)
        shear = np.random.uniform(*self.shear_range)
        translation = (
            np.random.uniform(*self.translation_range) * w,
            np.random.uniform(*self.translation_range) * h
        )
        af = AffineTransform(scale=scale, shear=shear, rotation=rotation, translation=translation)
        img_data1 = warp(img_data, af.inverse)
        img1 = Image.fromarray(np.uint8(img_data1 * 255))
        return img1