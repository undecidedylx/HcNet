# -- coding: utf-8 --
import os
from os.path import join
import parameter as paras
import random
import shutil

"""
4.目标路径TB_last
按比例划分数据集
"""
random.seed(10)

TB_nii_dir = paras.TB_seting_dir

train_dir = os.path.join(paras.last_data_dir + "/", 'train')
os.makedirs(train_dir, exist_ok=True)
val_dir = train_dir.replace('train', 'val')
os.makedirs(val_dir, exist_ok=True)
test_dir = train_dir.replace('train', 'test')
os.makedirs(test_dir, exist_ok=True)

for cls in os.listdir(TB_nii_dir):
    train_dest = join(train_dir, cls)
    os.makedirs(train_dest, exist_ok=True)
    test_dest = join(test_dir, cls)
    os.makedirs(test_dest, exist_ok=True)
    val_dest = join(val_dir, cls)
    os.makedirs(val_dest, exist_ok=True)

    print(train_dest)
    print(val_dest)

    cls_dir = join(TB_nii_dir, cls)
    nii_path_list = [join(cls_dir + '/', nii_file) for nii_file in os.listdir(cls_dir)]
    random.shuffle(nii_path_list)

    sample_num = len(nii_path_list)
    train_num = int(sample_num * 0.8)
    val_num = int(sample_num * 0.1)


    train_samples = nii_path_list[:train_num]
    val_samples = nii_path_list[train_num:train_num + val_num]
    test_samples = nii_path_list[train_num + val_num:]

    print(len(train_samples))
    print(len(val_samples))
    print(len(test_samples))

    for train_sample in train_samples:
        shutil.copy(train_sample, train_dest)

    for val_sample in val_samples:
        shutil.copy(val_sample, val_dest)

    for test_sample in test_samples:
        shutil.copy(test_sample, test_dest)
