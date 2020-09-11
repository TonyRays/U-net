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
# from utils import custom_transforms as tr
import custom_transforms as tr
import cv2


import torchvision.transforms as transforms

class LungDataset(Dataset):
    # mean and std
    # mean = 0.2792356
    # std = 0.30019337
    mean = 0.25409937
    std = 0.29017985


    def __init__(self, root_dir, transforms, train=True):
        self.root_dir = root_dir
        if train:
            self.image_folder = os.path.join(root_dir, 'train_images_1')
        else:
            self.image_folder = os.path.join(root_dir, 'test_images_1')
        # self.mask_folder = os.path.join(root_dir, 'mask')
        self.transforms = transforms
        self.train = train
        self.data, self.targets = self.get_data_targets()
        # self.mean , self.std = get_mean_std(self, ratio=0.1)

    def get_data_targets(self):
        '''

        :return:已知数据集的根目录 self.image_folder,返回每张图片路径以及对应的标签路径的数组
        '''
        image_folders = os.listdir(self.image_folder)
        data = []
        target = []
        for case in image_folders:
            case_path = os.path.join(self.image_folder,case)
            imgs_path = os.path.join(case_path,'img\\')
            # imgs_path = case_path
            print(imgs_path)
            img_files = os.listdir(imgs_path)
            masks_path = os.path.join(case_path,'mask\\')

            # for img_file in sorted(img_files, key = lambda x: int(x.split('.')[0])):
            for img_file in img_files:
                img_path = os.path.join(imgs_path, img_file)
                shotname, extension = os.path.splitext(img_file)
                # print(shotname)
                mask_path = os.path.join(masks_path, shotname + '_mask'  + extension)
                data.append(img_path)
                target.append(mask_path)
        # list_path = self.image_folder
        # g = os.walk(list_path)
        #
        # for path, d, filelist in g:
        #     for filename in filelist:
        #         if filename.endswith('jpg'):
        #             data.append(path + "\\" + filename)
        #             # print(os.path.join(path, filename))
        #
        # list_path = self.mask_folder
        # g = os.walk(list_path)
        #
        # for path, d, filelist in g:
        #     for filename in filelist:
        #         if filename.endswith('jpg'):
        #             target.append(path + "\\" +  filename)
        #
        # print(self.image_folder)
        # print(data)
        # print(len(data))
        # print(self.mask_folder)
        # print(target)
        # print(len(target))

        return data, target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''

        :param idx: 数组data和target的index
        :return: 返回经过transform处理后的图片和标签，sample={'image':img, 'label':label}
        '''
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        image_path = self.data[idx]
        target_path = self.targets[idx]

        # img =[]
        # label =[]
        # for path in self.data:
        #     img_temp = cv2.imread(path)
        #     img.append(img_temp)
        # for path in self.targets:
        #     label_temp = cv2.imread(path)
        #     label.append(label_temp)

        image = self.pil_loader(image_path)
        target = self.pil_loader(target_path)
        # image = cv2.imread(image_path)
        # target = cv2.imread(target_path)
        # img = cv2.imread(self.data)
        # label = cv2.imread(self.targets)

        sample = {'image': image, 'label': target}

        if self.transforms:
            sample = self.transforms(sample)

        sample = {'image': sample['image'].unsqueeze(dim=0), 'label': sample['label'].unsqueeze(dim=0)}
        # sample = {'image': sample['image'], 'label': sample['label']}

        # print(len(sample))
        return sample

    def pil_loader(self, file_path):
        image = Image.open(file_path)
        image_transforms = transforms.Compose([
            transforms.Grayscale(1)
        ])
        image = image_transforms(image)
        return image

    def get_mean_std(self, ratio=0.1):
        trs = tf.Compose([
            tr.FixedResize(512),
            tr.Normalize(mean=0, std=1),
            tr.ToTensor()
        ])
        dataset = LungDataset(root_dir=r'D:\code\U-net', transforms=trs, train=True)
        print(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                                 shuffle=True, num_workers=4)

        for item in dataloader:
            train = item['image']
            # train = np.array(train)      #?
            print(train.shape)
            print('sample {} images to calculate'.format(train.shape[0]))
            mean = np.mean(train.numpy(), axis=(0, 2, 3))
            std = np.std(train.numpy(), axis=(0, 2, 3))
        return mean, std


if __name__ == '__main__':
    trs = tf.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomScaleCrop(base_size=512, crop_size=512),
        tr.RandomGaussianBlur(),    #高斯模糊
        tr.Normalize(mean=LungDataset.mean, std=LungDataset.std),
        tr.ToTensor()
    ])
    dataset = LungDataset(root_dir=r'D:\code\U-net', transforms=trs, train=True)
    # dataset = LungDataset(root_dir=r'D:\code\U-net', transforms = False , train=True)
    # print(dataset.get_mean_std())
    for item in dataset:
        # print(item['label'].shape)
        # plt.imshow(image, cmap='gray')
        # print(item['image'])
        print(item['image'].min(), item['image'].max(), item['label'].min(), item['label'].max())