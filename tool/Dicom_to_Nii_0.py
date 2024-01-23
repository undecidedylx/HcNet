# -- coding: utf-8 --
import os
import numpy as np
import pydicom
import nibabel
from scipy.ndimage import zoom



def __affine(vol_sz):
    """
    利用像素体积构建仿射矩阵
    :param vol_sz: 整个序列的像素体积信息
    """
    return [[vol_sz[0], 0, 0, 0],
            [0, vol_sz[1], 0, 0],
            [0, 0, vol_sz[2], 0],
            [0, 0, 0, 1]]


def read_dicom(dcm_dir):
    """
    将整个 dcm序列转化为一个nii文件

    :param dcm_dir:一个dcm序列的路径
    :param nii_dir: 整个dcm序列转为一个nii 的保存路径
    :param volume_name: nii文件的命名
    """
    # 获取所有的dcm文件名
    dcms = os.listdir(dcm_dir)
    first_dcm = pydicom.read_file(os.path.join(dcm_dir, dcms[0]))

    # 获取整个序列的像素体积
    voxel_size = np.array([first_dcm.PixelSpacing[0], first_dcm.PixelSpacing[1], first_dcm.SliceThickness])
    # 创建数组
    data = np.zeros(np.array([512, 512, len(dcms)], dtype=int))

    # 遍历处理每个dcm文件
    for index, d_file in enumerate(dcms):
        process_text = f'process data: {index + 1} ({(index + 1) / len(dcms) * 100:.2f}%)'
        print('\r', process_text, end='', flush=True)

        # 读取每个dcm数据
        d_info = pydicom.read_file(os.path.join(dcm_dir, d_file))
        # 提取像素信息 并将像素值转化为Hu值
        array = d_info.pixel_array
        array = array * float(d_info.RescaleSlope) + float(d_info.RescaleIntercept)

        # 调整尺寸
        target_size = (512, 512)

        # 使用 zoom 函数进行形状调整
        array = zoom(array, (target_size[0] / array.shape[0], target_size[1] / array.shape[1]))
        #
        #         print()
        #         print(data.shape, array.shape)
        #         print(d_info.Rows, d_info.Columns)
        #         # 按照计算的depth维度的索引存如data中
        data[:, :, index] = array

    return data, voxel_size


#
def __affine(vol_sz):
    return [[vol_sz[0], 0, 0, 0],
            [0, vol_sz[1], 0, 0],
            [0, 0, vol_sz[2], 0],
            [0, 0, 0, 1]]


#
#
def read_store(dcm_dir, nii_dir, vloume_name):
    try:
        data, voxel_size = read_dicom(dcm_dir)
        if not os.path.exists(nii_dir):
            os.makedirs(nii_dir)
        nii_file = nibabel.Nifti1Image(data, __affine(voxel_size))
        print(os.path.join(nii_dir, vloume_name))
        nii_file.to_filename(os.path.join(nii_dir, vloume_name))
    except:
        print(f"worrong:{dcm_dir}")


if __name__ == '__main__':
    dcm_normal_dir = 'G:/TB_row/Non-Active/丁成芬_GYCT10380123/1.3.6.1.4.1.46677.102.869006.10380123.2208302329/1.3.12.2.1107.5.1.4.60464.30000022082923532416000180087'
    nii_dir = r'../out_put/'
    volume_normal = 'test.nii'
    read_store(dcm_normal_dir, nii_dir, volume_normal)
