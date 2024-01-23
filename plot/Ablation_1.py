# -- coding: utf-8 --


import pandas as pd
import matplotlib.pyplot as plt

markers = ['D', 's', 'o', 'x', '^', 'p']
file_path = 'input_resolutions.xls'
df = pd.read_excel(file_path)

plt.figure(figsize=(9, 5))
for index, (row, marker) in enumerate(zip(df.iterrows(), markers)):
    plt.plot(df.columns[1:], row[1][1:], marker=marker, label=row[1]['Model'],linewidth=2.5,markersize=10)
# 设置图表标题和标签

plt.xlabel('Image Resolution',fontsize=14)
plt.ylabel('Accuracy (%)',fontsize=14)
plt.legend(loc='best')  # 显示图例

# 显示图表
plt.show()