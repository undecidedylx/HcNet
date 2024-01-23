# -- coding: utf-8 --
from torch.utils.data import DataLoader
from dataset.dataset import Dataset
# from model.ResNet_3D import ResNet_3d, Bottleneck
# from model.DenseNet_3D import DenseNet_3d
# from model.Efficient_3D import EfficientNet_b0_3d
from model.HcNet import HcNet
import torch
import parameter as paras
import os
import json
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix, classification_report

def loss_adc_acc():
    # df = pd.read_csv(r'/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/result/self_weight/loss_acc_ResNet.csv')
    # df = pd.read_csv(r'/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/result/self_weight/loss_acc_DenseNet_3D.csv')
    df = pd.read_csv(
        r'/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/result/self_weight/loss_acc_EfficientNet_3D.csv')
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
        num_classes = self.num_classes
        labels = self.labels
        return matirx, num_classes, labels
        # plt.imshow(matirx, cmap=plt.cm.Blues)
        # # 设置x轴坐标label
        # plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # # 设置y轴坐标label
        # plt.yticks(range(self.num_classes), self.labels)
        #
        # # 显示colorbar
        # plt.colorbar()
        # plt.xlabel('True Labels')
        # plt.ylabel('Predicted Labels')
        # plt.title('EfficientNet_3D_Confusion matrix')
        # # 在图中标注数量/概率信息
        # thresh = matirx.max() / 2
        # for x in range(self.num_classes):
        #     for y in range(self.num_classes):
        #         # 注意这里的matrix[y, x]不是matrix[x, y]
        #         info = int(matirx[y, x])
        #         plt.text(x, y, info,
        #                  verticalalignment='center',
        #                  horizontalalignment='center',
        #                  color="white" if info > thresh else "black")
        # plt.tight_layout()
        # plt.savefig('/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/reslut_img/EfficientNet_3D_ConfusionMatrix.png')
        # plt.show()
        # plt.close()


def predict_model(model_name, model, m_list, fpr_list, tpr_list, auc_list):
    print(f"model:{model_name} predicting---------------")
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    nw = min(os.cpu_count(), paras.batch_size if paras.batch_size > 1 else 0, 8)

    model_weight_path = os.path.join(paras.save_path, f'{model_name}.pth')
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)

    json_label_path = '../class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=2, labels=labels)
    model.eval()

    # 预测
    if model_name == 'HcNet':
        test_dataset = Dataset(nii_dir=r'/remote-home/hongzhangxin/pytorch_project/My_Data/TB_data/test_selecet')
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=paras.batch_size, num_workers=nw,
                                 pin_memory=True)
        predicted_labels = []
        y_true = []
        y_scores = []
        with torch.no_grad():
            for val_data in tqdm(test_loader):
                val_images, val_labels, name = val_data
                outputs = model(val_images.to(device))
                out = outputs[0] + outputs[1]
                out_prob = torch.softmax(out, dim=1)
                out = torch.argmax(out_prob, dim=1)
                confusion.update(out.to("cpu").numpy(), val_labels.to("cpu").numpy())
                for x in val_labels.cpu().numpy():
                    y_true.append(x)
                for y in out_prob[:, 1].cpu().numpy():
                    y_scores.append(y)
                for p in out.cpu().numpy():
                    predicted_labels.append(p)

            confusion.summary()
            m, n, l = confusion.plot()
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
            conf_matrix = confusion_matrix(y_true, predicted_labels)
            class_report = classification_report(y_true, predicted_labels, target_names=['Non-Active', 'Active'],
                                                 labels=[0, 1], output_dict=True)['weighted avg']
            print("AUC：", auc)
            print("\nClassification Report:")
            print(class_report)

            m_list.append(m)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(auc)




    else:
        test_dataset = Dataset(nii_dir=os.path.join(paras.last_data_dir, 'test'))
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=paras.batch_size, num_workers=nw,
                                 pin_memory=True)
        y_true = []
        y_scores = []
        predicted_labels = []
        with torch.no_grad():
            for val_data in tqdm(test_loader):
                val_images, val_labels, name = val_data
                outputs = model(val_images.to(device))
                out_prob = torch.softmax(outputs, dim=1)
                outputs = torch.argmax(out_prob, dim=1)
                confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
                for x in val_labels.cpu().numpy():
                    y_true.append(x)
                for y in out_prob[:, 1].cpu().numpy():
                    y_scores.append(y)
                for p in outputs.cpu().numpy():
                    predicted_labels.append(p)

        confusion.summary()
        m, n, l = confusion.plot()
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        print("AUC：",auc)
        m_list.append(m)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)

        conf_matrix = confusion_matrix(y_true, predicted_labels)
        class_report = classification_report(y_true, predicted_labels, target_names=['Non-Active', 'Active'],
                                             labels=[0, 1], output_dict=True)['weighted avg']

        print("\nClassification Report:")
        print(class_report)
    return m_list,fpr_list,tpr_list,auc_list


if __name__ == '__main__':
    # model_resnet = ResNet_3d(Bottleneck, [3, 4, 6, 3], shortcut_type='B', no_cuda=False, num_classes=2,
    #                          include_top=True)
    # model_densenet = DenseNet_3d(num_classes=2, num_init_features=64, growth_rate=32, block_config=(6, 12, 64, 48))
    # model_effnet = EfficientNet_b0_3d(width_coefficient=1.0,
    #                                   depth_coefficient=1.0,
    #                                   dropout_rate=0.2,
    #                                   num_classes=2)
    model_HcNet = HcNet(num_classes=2, K=16, group_num=50)

    model_name_list = ['HcNet', 'ResNet', 'DenseNet', 'EfficientNet']
    model_list = [model_HcNet]

    m_list = []
    fpr_list = []
    tpr_list = []
    auc_list = []
    linestyles = ['-', '--', '-.', ':']

    for i in range(4):
        predict_model(model_name=model_name_list[i], model=model_list[i], m_list=m_list,
                      fpr_list=fpr_list, tpr_list=tpr_list, auc_list=auc_list)


    plt.figure(figsize=(8, 6))
    # 绘制第一个模型的ROC曲线
    plt.plot(fpr_list[0], tpr_list[0], color='b', lw=2,linestyle=linestyles[0], label=f'{model_name_list[0]} (AUC = {auc_list[0]:.2f})',)

    # 绘制第二个模型的ROC曲线
    plt.plot(fpr_list[1], tpr_list[1], color='g', lw=2,linestyle=linestyles[0], label=f'{model_name_list[1]} (AUC = {auc_list[1]:.2f})')

    # 绘制第三个模型的ROC曲线
    plt.plot(fpr_list[2], tpr_list[2], color='r', lw=2,linestyle=linestyles[0], label=f'{model_name_list[2]} (AUC = {auc_list[2]:.2f})')

    # 绘制第四个模型的ROC曲线
    plt.plot(fpr_list[3], tpr_list[3], color='c', lw=2,linestyle=linestyles[0], label=f'{model_name_list[3]} (AUC = {auc_list[3]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/reslut_img/Receiver Operating Characteristic (ROC) Curve.png')
    plt.show()

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    # 绘制每个模型的混淆矩阵
    print(type(m_list))

    # 创建一个大图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 创建一个大图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 循环遍历每个模型的混淆矩阵并绘制
    for i, m in enumerate(m_list):
        row, col = divmod(i, 2)
        disp = ConfusionMatrixDisplay(confusion_matrix=m, display_labels=['Non-Active', 'Active'])
        disp.plot(cmap=plt.cm.Blues, ax=axes[row, col])
        axes[row, col].set_title(model_name_list[i])
    # 调整子图布局
    plt.tight_layout()
    # 显示图形
    plt.savefig('/remote-home/hongzhangxin/pytorch_project/TB_Classfiy/reslut_img/Confusion Matrix.png')
    plt.show()