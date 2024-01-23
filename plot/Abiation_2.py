import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

# 读取XLS文件
file_path = 'DSP_result.xls'
df = pd.read_excel(file_path)

# 获取模型名称和K的取值
models = df['Model'].unique()
k_values = df.columns[1:]

# 设置柱状图的宽度
bar_width = 0.2

# 生成位置数组
positions = np.arange(len(k_values))

# 定义浅色颜色
light_colors = ['lightblue', 'lightyellow', 'lightcyan', 'lightpink']
dark_colors = ['darkblue', 'darkorange', 'darkcyan', 'darkred']
markers = ['D', 's', 'o', 'x']

# 绘制柱状图
plt.figure(figsize=(12, 8))

for i, (model, color) in enumerate(zip(models, light_colors)):
    model_data = df[df['Model'] == model]
    plt.bar(positions + i * bar_width, model_data.iloc[:, 1:].values.flatten(), width=bar_width, label=model,
            color=color)

# 设置y轴范围
plt.ylim(0.75, 1)

# 设置图表标题和标签

plt.xlabel('Different values of hyperparameter K in the DSP module',fontsize=14)
plt.ylabel('Accuracy (%)',fontsize=14)

# 绘制折线图
for i, (model, dark_color, marker) in enumerate(zip(models, dark_colors, markers)):
    model_data = df[df['Model'] == model]

    # 计算深色颜色（颜色深度减小）
    mid_color = to_rgba(dark_color, alpha=0.35)

    plt.plot(positions + i * bar_width, model_data.iloc[:, 1:].values.flatten(), marker=marker, markersize=10,
             linestyle='--', color=mid_color, label=f'{model} Trend', linewidth=3.5)  # 调整linewidth参数

plt.xticks(positions + bar_width * (len(models) - 1) / 2, k_values, rotation=45)  # 设置X轴刻度位置和标签，并进行45度旋转
plt.legend(loc='best')  # 显示图例

# 显示图表
plt.show()
