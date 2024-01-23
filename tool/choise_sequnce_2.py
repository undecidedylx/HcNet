# -- coding: utf-8 --
from os.path import join
import os
import shutil
import parameter as paras

"""
2.目标路径 TB
选取每个患者的需要的序列
"""
nii_dir = paras.nii_dir
TB_dir = paras.TB_dir

for class_ in os.listdir(nii_dir):
    print(class_)
    class_path = join(nii_dir + '/', class_)
    patient_list = [join(class_path + '/', patient) for patient in os.listdir(class_path)]
    for patient in patient_list:
        nii_file_path = [join(patient + '/', nii_file) for nii_file in os.listdir(patient)]
        select_file_path = [path for path in nii_file_path if path.endswith('-0.nii')]

        for volume_path in select_file_path:
            out_path = join(TB_dir + '/', class_)
            os.makedirs(out_path, exist_ok=True)
            print(volume_path)
            print(out_path)
            shutil.copy(volume_path, out_path)
    print("**" * 30)
