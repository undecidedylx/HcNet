# -- coding: utf-8 --
import os
from os.path import join
import parameter as paras
import SimpleITK as sitk
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

"""
3.目标路径 TB_seting 
处理nii数据 如设置窗口 统一厚度，长宽 选择合适的slices
"""


def con_plot(ct_array1, ct_array_windows):
    for i in range(ct_array.shape[0]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(ct_array_windows[i], cmap='gray')
        axes[0].set_title(f"CT with Windowing{i}")

        axes[1].imshow(ct_array[i], cmap='gray')
        axes[1].set_title(f"Original CT{i}")
        # 调整图像布局
        plt.tight_layout()
        # 显示图像
        plt.show()

list_count = []
count_num = 0
TB_dir = paras.TB_dir
TB_seting = paras.TB_seting_dir

for cls_ in os.listdir(TB_dir):
    save_path = os.path.join(TB_seting, cls_)

    os.makedirs(save_path, exist_ok=True)
    cls_path = join(TB_dir, cls_)
    nii_file_list = [join(cls_path + '/', file_name) for file_name in os.listdir(cls_path)]
    for nii_file in nii_file_list:
        print(nii_file)
        file_name = nii_file.split('/')[-1]
        ct = sitk.ReadImage(nii_file)
        ct_array = sitk.GetArrayFromImage(ct)
        print(ct)
        # 设置窗口中心和窗口宽度
        window_center = paras.window_center  # 根据需求进行调整
        window_width = paras.window_width  # 根据需求进行调整
        # print(ct_array.shape)
        # print(np.max(ct_array),np.min(ct_array),np.average(ct_array))
        ct_array_windows = np.clip((ct_array - (window_center - 0.5)) / (window_width - 1.0) + 0.5, 0, 1)

        # 对CT数据在横断面上进行降采样,并进行重采样,将所有数据的z轴的spacing调整到5mm
        ct_resample = nd.zoom(ct_array_windows, (ct.GetSpacing()[-1] / paras.slice_thickness,
                                                 paras.down_scale, paras.down_scale), order=3)
        print(ct_resample.shape)


        slices_num, H, W = ct_resample.shape


        if slices_num <= 50:
            list_count.append(nii_file.split('/')[-1])
            print(nii_file.split('/')[-1])
            print("小于50")
            target_depth = 50
            zoom_factor = target_depth / slices_num
            ct_resample = zoom(ct_resample, (zoom_factor, 1, 1), order=3)

        else:
            mid = ct_resample.shape[0] // 2
            ct_resample = ct_resample[mid - 25:mid + 25, :, :]

        new_ct = sitk.GetImageFromArray(ct_resample)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(
            (ct.GetSpacing()[0] * int(1 / paras.down_scale), ct.GetSpacing()[1] * int(1 / paras.down_scale),
             paras.slice_thickness))
        array = sitk.GetArrayFromImage(new_ct)
        print(array.shape)
        sitk.WriteImage(new_ct, os.path.join(save_path + '/', file_name))
        print(os.path.join(save_path, file_name))
        print("--------------------------------------------------------------------------")

print(list_count)