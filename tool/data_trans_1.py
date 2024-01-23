# -- coding: utf-8 --
import os
from os.path import join
from tool.Dicom_to_Nii_0 import read_store
import parameter as paras

"""
1.目标路径 TB_nii
将每个患者的每个序列下所有的dcm 转换为一个nii
命名： volume-{患者编号}-{序列编号}.nii
"""


def main(dicom_dir, nii_dir):
    dicom_class_list = [join(dicom_dir, class_ + '/') for class_ in os.listdir(dicom_dir)]
    patien_number = 233
    # # 每个类别
    for i, class_path in enumerate(dicom_class_list):
        class_name = class_path.split('/')[-2]
        if class_name == "Active":
            continue
        # 获取每个类别下的患者的路径
        class_patien_list = [join(class_path, patien_ + '/') for patien_ in os.listdir(class_path)]
        for one_patien in class_patien_list:
            # 每个患者的目录
            patien_dir = join(one_patien, os.listdir(one_patien)[0] + '/')

            # 每个患者的序列目录
            patien_sequence = [join(patien_dir, sequence) for sequence in os.listdir(patien_dir)]
            sequence_number = 0

            for index, sequence in enumerate(patien_sequence):
                if sequence_number >= 5:
                    continue
                dcm_num = len(os.listdir(sequence))
                if dcm_num < 30 or dcm_num > 100:
                    continue
                # 构建nii文件名 volume-{patien_number}-{sequence_number}.nii
                patient_number = "patient_{}".format(patien_number)
                nii_file_name = 'volume-{}-{}.nii'.format(patien_number, sequence_number)
                print(f"patien_number:0{patien_number}------sequence_number:{sequence_number}------"
                      f"dicom_num:{dcm_num}")

                # 构建nii路径
                class__ = class_path.split('/')[-2]
                nii_file_path = join(nii_dir, class__ + '/')
                nii_file_path = join(nii_file_path + '/', patient_number)
                print(sequence)
                print(nii_file_path)
                print(nii_file_name)
                # if patien_number >= 70:
                read_store(dcm_dir=sequence, nii_dir=nii_file_path, vloume_name=nii_file_name)
                sequence_number += 1
                print("----------------------------------------------------------------")
            patien_number += 1


if __name__ == '__main__':
    Dicom_dir = paras.dicom_dir
    Nii_dir = paras.nii_dir
    main(Dicom_dir, Nii_dir)
