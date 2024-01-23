# -- coding: utf-8 --
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from model.Deep_Slice_Prioritizer import DSP
import torch
import torch.nn as nn

def all_plot(ct_array):
    # 读取ct_array的形状
    num_slices, height, width = ct_array.shape

    # 设置每个图中要显示的切片数量
    slices_per_figure = 30

    # 间距设置
    hspace = 0.05  # 垂直间距
    wspace = 0.05  # 水平间距

    # 计算需要多少个图
    num_figures = (num_slices - 1) // slices_per_figure + 1

    for figure_num in range(num_figures):
        start_slice = figure_num * slices_per_figure
        end_slice = min(start_slice + slices_per_figure, num_slices)
        num_rows = (end_slice - start_slice - 1) // 5 + 1
        num_cols = min(end_slice - start_slice, 5)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 30))

        for i in range(start_slice, end_slice):
            row = (i - start_slice) // 5
            col = (i - start_slice) % 5
            ax = axes[row, col]
            img = ct_array[i]
            # print(f"slice:{i}")
            # print("max", np.max(img))
            # print("min", np.min(img))
            # print("mean", np.mean(img))
            # print("std", np.std(img))
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Slice: {i}")

        # 删除多余的子图
        for i in range(end_slice - start_slice, num_rows * num_cols):
            row = i // 5
            col = i % 5
            fig.delaxes(axes[row, col])

        # 调整布局
        plt.tight_layout(h_pad=hspace, w_pad=wspace)

        plt.show()


path = r"volume-8-0.nii"
ct = sitk.ReadImage(path)
ct_array = sitk.GetArrayFromImage(ct)
# all_plot(ct_array)
ct_tensor = torch.Tensor(ct_array)
print(ct_tensor.shape)
# deep_select = DSP()
# out = deep_select(ct_array)



