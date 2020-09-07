from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
# from monai import transforms as mn_tf
from tqdm import tqdm
import os
from torchvision import transforms as tf
import torch.utils.data as Data
from utils import custom_transforms as tr


class LungDataset(Dataset):
    # mean and std
    mean =
    std =

    def __init__(self, root_dir, transforms, train=True):
        self.root_dir = root_dir
        if train:
            self.image_folder = os.path.join(root_dir, 'train_images_1')
        else:
            self.image_folder = os.path.join(root_dir, 'test_images_1')
        self.transforms = transforms
        self.train = train
        self.data, self.targets = self.get_data_targets()

    def get_data_targets(self):
        '''

        :return:已知数据集的根目录 self.image_folder,返回每张图片路径以及对应的标签路径的数组
        '''
        data = []
        target = []
        list_path = self.image_folder

        g = os.walk(list_path)
        img_files = ['%s\\%s' % (i[0], j) for i in g for j in i[-1] if

                     '''
            TODO
        '''
        return data, target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''

        :param idx: 数组data和target的index
        :return: 返回经过transform处理后的图片和标签，sample={'image':img, 'label':label}
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)



        '''
            TODO
        '''

        return sample

    def pil_loader(self, file_path):
        return Image.open(file_path)

    def get_mean_std(self, ratio=0.1):
        trs = tf.Compose([
            tr.FixedResize(512),
            tr.Normalize(mean=0, std=1),
            tr.ToTensor()
        ])
        dataset = LungDataset(root_dir='../../../dataset/LUNA16', transforms=trs, train=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                                 shuffle=True, num_workers=4)
        for item in dataloader:
            train = item['image']
            print(train.shape)
            print('sample {} images to calculate'.format(train.shape[0]))
            mean = np.mean(train.numpy(), axis=(0, 2, 3))
            std = np.std(train.numpy(), axis=(0, 2, 3))
            return mean, std


if __name__ == '__main__':
    trs = tf.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomScaleCrop(base_size=512, crop_size=512),
        tr.RandomGaussianBlur(),
        tr.Normalize(mean=LungDataset.mean, std=LungDataset.std),
        tr.ToTensor()
    ])
    dataset = LungDataset(root_dir='../../../dataset/LUNA16', transforms=trs, train=True)
    # print(dataset.get_mean_std())
    for item in dataset:
        # print(item['image'].shape)
        print(item['image'].min(), item['image'].max(), item['label'].min(), item['label'].max())