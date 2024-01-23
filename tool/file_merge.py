# -- coding: utf-8 --

"""
合并 同一个人的所有zip 到 一个新的文件，命名为 姓名_住院号
"""
import os
import shutil
import pandas as pd

import os
import shutil
"""
合并 同一个人的zip 到 一个新的文件，命名为 姓名_住院号
"""
# root_directory = "E:/work/My_Data/TB/Activate/"
root_directory = "E:/work/My_Data/TB/Non-Activate/"

# 获取根目录下所有的子文件列表
subfiles = [filename for filename in os.listdir(root_directory) if os.path.isfile(os.path.join(root_directory, filename)) and filename.endswith('.zip')]
print(subfiles)

# 创建一个文件夹用于保存合并后的子文件
# output_folder = r'E:/work/My_Data/TB_row/Activate/'
output_folder = r'E:/work/My_Data/TB_row/Non-Activate/'
os.makedirs(output_folder, exist_ok=True)

# 读取excel
# df = pd.read_excel('../result_preprocess/tb_secondary.xlsx')
df = pd.read_excel('../result_preprocess/tb_ancient.xlsx')


# 遍历每个子文件，生成字典 {姓名：[对应姓名的.zip文件，....]}
name_files = {}
for subfile in subfiles:
    # 获取姓名
    name_key = subfile.split('_')[0]
    if name_key not in name_files:
        name_files[name_key] = []
    name_files[name_key].append(subfile)


for name_key, files in name_files.items():
    filtered_rows = df[df['姓名'] == name_key]
    # 获取住院号
    hospital_numbers = filtered_rows['住院号'].tolist()[0]
    # 创建文件  姓名_住院号
    name_folder = os.path.join(output_folder, f"{name_key}_{hospital_numbers}")
    os.makedirs(name_folder, exist_ok=True)

    for subfile in files:
        source_path = os.path.join(root_directory,subfile)
        destination_path = name_folder
        print(source_path)
        print(destination_path)
        print("----------")
        shutil.move(source_path,destination_path)
#
print("合并完成！")