# -- coding: utf-8 --
import os
from os.path import join
from torch.utils.data import Dataset as dataset
from torch.utils.data import DataLoader
import SimpleITK as sitk
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.ndimage as ndimage
from scipy.ndimage import rotate
import parameter as paras
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ToPILImage
import random


class Dataset(dataset):
    def __init__(self, nii_dir, transform=None):
        super().__init__()

        self.data_path_list = []
        self.label_list = []
        self.transform = transform
        class_dir = [join(nii_dir + '/', class_) for class_ in os.listdir(nii_dir)]
        for cla in class_dir:
            if cla.split('/')[-1] == 'Active':
                label = 1
            else:
                label = 0

            for nii_file in os.listdir(cla):
                nii_path = join(cla + '/', nii_file)
                self.data_path_list.append(nii_path)
                self.label_list.append(label)

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, index):
        data_path = self.data_path_list[index]
        ct = sitk.ReadImage(data_path)
        ct_array = sitk.GetArrayFromImage(ct)
        label = self.label_list[index]
        volume_name = self.data_path_list[index].split('/')[-1]

        ct_tensor = torch.Tensor(ct_array)
        if self.transform:
            ct_tensor = self.transform(ct_tensor)


        return ct_tensor.unsqueeze(0), label, volume_name






if __name__ == '__main__':
    transform = transforms.Compose([
        RandomHorizontalFlip(),
        RandomRotation(degrees=90)

    ])

    train_data_dir = os.path.join(paras.last_data_dir, 'train')
    train_dataset = Dataset(train_data_dir, transform=transform)
    data,label,name  = train_dataset.__getitem__(index=400)
    print(data)
    print(torch.max((data)))
    print(torch.min((data)))

    # train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=1)
    # for data in train_loader:
    #     ct_array, label, name = data
    #     img_ = ct_array[0]
    #     img = img_[0]
    #     print(img.shape,label,name)
    #     for i in range(img.shape[0]):
    #         plt.imshow(img[i],cmap='gray')
    #         plt.show()

