# -- coding: utf-8 --
import os
import random
import shutil
from os.path import join
data_row = r'E:\work\My_Data\archive_row'
data_new = r'E:\work\My_Data\archive'

train_dir = os.path.join(data_new + "/", 'train')
os.makedirs(train_dir, exist_ok=True)
val_dir = train_dir.replace('train', 'val')
os.makedirs(val_dir, exist_ok=True)
test_dir = train_dir.replace('train', 'test')
os.makedirs(test_dir, exist_ok=True)


for cls in os.listdir(data_row):
    train_dest = join(train_dir + '/', cls)
    os.makedirs(train_dest, exist_ok=True)
    test_dest = join(test_dir + '/', cls)
    os.makedirs(test_dest, exist_ok=True)
    val_dest = join(val_dir + '/', cls)
    os.makedirs(val_dest, exist_ok=True)

    cls_dir = join(data_row, cls)

    data_path_list = [join(cls_dir + '/', data_file) for data_file in os.listdir(cls_dir)]

    random.shuffle(data_path_list)
    sample_num = len(data_path_list)
    train_num = int(sample_num * 0.8)
    val_num = int(sample_num * 0.1)

    train_samples = data_path_list[:train_num]
    val_samples = data_path_list[train_num:train_num + val_num]
    test_samples = data_path_list[train_num + val_num:]

    print(len(train_samples))
    print(len(val_samples))
    print(len(test_samples))

    for train_sample in train_samples:
        shutil.copy(train_sample, train_dest)

    for val_sample in val_samples:
        shutil.copy(val_sample, val_dest)

    for test_sample in test_samples:
        shutil.copy(test_sample, test_dest)