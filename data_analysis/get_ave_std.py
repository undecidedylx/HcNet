# -- coding: utf-8 --
import os
import SimpleITK as sitk
from tqdm import tqdm
import parameter as para
import numpy as np
from torchvision import transforms

"""
统计Z 轴方向上的 平均厚度
"""

nii_ave_list = []
nii_std_list = []
spacing_list = []
data_dir = para.last_data_dir + 'train'
for cls_ in os.listdir(data_dir):
    cls_path = os.path.join(data_dir, cls_)
    nii_path_list = [os.path.join(cls_path + '/', patient) for patient in os.listdir(cls_path)]
    for nii_file in nii_path_list:
        ct = sitk.ReadImage(nii_file)
        ct_array = sitk.GetArrayFromImage(ct)
        print(np.average(ct_array), np.max(ct_array), np.min(ct_array))
        nii_ave_list.append(np.average(ct_array))
        nii_std_list.append(np.std(ct_array))
        spacing_list.append(ct.GetSpacing()[-1])

print(f"spacing is {sum(spacing_list) / len(spacing_list)}")
print(f"ave is {sum(nii_ave_list) / len(nii_ave_list)}")
print(f"std is {sum(nii_std_list) / len(nii_std_list)}")

# spacing is 4.7
# ave is -562.1964923035515
# std is 482.9896570342814



# spacing is 5.0
# ave is 0.3657577251109106
# std is 0.31137916367230756