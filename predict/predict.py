# -- coding: utf-8 --
from torch.utils.data import DataLoader
from dataset.dataset import Dataset

import torch
import parameter as paras
import os
import json
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from tqdm import tqdm


def loss_adc_acc():
    # df = pd.read_csv(r'/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/result/self_weight/loss_acc_ResNet.csv')
    # df = pd.read_csv(r'/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/result/self_weight/loss_acc_DenseNet_3D.csv')
    df = pd.read_csv(r'/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/result/self_weight/loss_acc_EfficientNet_3D.csv')
    losses = df["loss"].values
    accuracies = df["acc"].values
    plt.title("DenseNet_3D_acc_loss")
    plt.plot(losses, label="Loss")
    plt.plot(accuracies, label="Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()

    plt.savefig("/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/reslut_img/acc_loss_EfficientNet_3D.png")
    plt.show()
    plt.close()


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        # 创建 全零的混淆矩阵
        self.matirx = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    # 填充混淆矩阵
    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            # 行是预测值 列是真实值
            self.matirx[p, t] += 1

    # 计算得到各项指标
    def summary(self):
        # ACC
        sum_TP = 0
        for i in range(self.num_classes):
            # 预测正确的是对角线之和
            sum_TP += self.matirx[i, i]
        acc = sum_TP / np.sum(self.matirx)
        print("the model accuracy is :", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "precision", "recall", "specificity"]
        for i in range(self.num_classes):
            TP = self.matirx[i, i]
            FP = np.sum(self.matirx[i, :]) - TP
            FN = np.sum(self.matirx[:, i]) - TP
            TN = np.sum(self.matirx) - TP - FP - FN
            precision = round(TP / (TP + FP), 3)
            recall = round(TP / (TP + FN), 3)
            specificity = round(TP / (TN + FP))
            table.add_row([self.labels[i], precision, recall, specificity])
        print(table)

    def plot(self):
        matirx = self.matirx
        print(matirx)
        plt.imshow(matirx, cmap=plt.cm.Blues)
        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('EfficientNet_3D_Confusion matrix')
        # 在图中标注数量/概率信息
        thresh = matirx.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matirx[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig('/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/reslut_img/EfficientNet_3D_ConfusionMatrix.png')
        plt.show()
        plt.close()


if __name__ == '__main__':
    loss_adc_acc()
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    nw = min(os.cpu_count(), paras.batch_size if paras.batch_size > 1 else 0, 8)
    test_dataset = Dataset(nii_dir=os.path.join(paras.last_data_dir, 'test'))
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=paras.batch_size, num_workers=nw,
                             pin_memory=True)

    json_label_path = '../class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    print(class_indict)

    # 加载模型 加载训练好的参数
    # model = DenseNet_3d(num_classes=2, num_init_features=64, growth_rate=32, block_config=(6, 12, 64, 48))
    model = EfficientNet_b0_3d(width_coefficient=1.0,
                                         depth_coefficient=1.0,
                                         dropout_rate=0.2,
                                         num_classes=2)
    # /remote-home/hongzhangxin/pytorch_project/TB_Classfiy/result/self_weight
    model_weight_path = os.path.join(paras.save_path, 'EfficientNet_3D.pth')
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)

    # 获取标签
    labels = [label for _, label in class_indict.items()]
    print(labels)
    confusion = ConfusionMatrix(num_classes=2, labels=labels)
    model.eval()

    with torch.no_grad():
        for val_data in tqdm(test_loader):
            val_images, val_labels, name = val_data
            outputs = model(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            print(val_labels)
            print(outputs)
            print(name)
            print("==============================")
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()
