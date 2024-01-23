import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = pd.read_excel("ex_result.xls")

# 提取模型名和指标数据
models = data["Model"]
acc = data["ACC"]
f1 = data["F1"]
pre = data["Pre"]
auc = data["AUC"]

# 创建一个自定义的颜色映射
custom_cmap = plt.cm.get_cmap("Set3", len(models))

# 创建一个2x2的子图布局
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# 定义一个辅助函数，用于在柱状图顶部添加数值标签
def add_labels(ax, data):
    for i, val in enumerate(data):
        ax.annotate(f'{val:.4f}', (i, val), ha='center', va='bottom', fontsize=8)

# 绘制ACC条形图
for i in range(len(models)):
    bars = axes[0, 0].bar(models[i], acc[i], color=custom_cmap(i), alpha=0.7)
    add_labels(axes[0, 0], acc)

axes[0, 0].set_title('ACC')
axes[0, 0].set_ylim(0, 1)
axes[0, 0].set_xticks(range(len(models)))
axes[0, 0].set_xticklabels(models, rotation=45, ha="right")

# 绘制F1条形图
for i in range(len(models)):
    bars = axes[0, 1].bar(models[i], f1[i], color=custom_cmap(i), alpha=0.7)
    add_labels(axes[0, 1], f1)

axes[0, 1].set_title('F1')
axes[0, 1].set_ylim(0, 1)
axes[0, 1].set_xticks(range(len(models)))
axes[0, 1].set_xticklabels(models, rotation=45, ha="right")

# 绘制Precision条形图
for i in range(len(models)):
    bars = axes[1, 0].bar(models[i], pre[i], color=custom_cmap(i), alpha=0.7)
    add_labels(axes[1, 0], pre)

axes[1, 0].set_title('Pre')
axes[1, 0].set_ylim(0, 1)
axes[1, 0].set_xticks(range(len(models)))
axes[1, 0].set_xticklabels(models, rotation=45, ha="right")

# 绘制AUC条形图
for i in range(len(models)):
    bars = axes[1, 1].bar(models[i], auc[i], color=custom_cmap(i), alpha=0.7)
    add_labels(axes[1, 1], auc)

axes[1, 1].set_title('AUC')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_xticks(range(len(models)))
axes[1, 1].set_xticklabels(models, rotation=45, ha="right")

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
