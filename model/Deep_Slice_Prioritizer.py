# -- coding: utf-8 --
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
import torch.nn.functional as F


class DeepNormal3D(nn.Module):
    def __init__(self,
                 group_num: int = 50,
                 eps: float = 1e-10,
                 K: int = 32):
        super().__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(group_num, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(group_num, 1, 1, 1))
        self.eps = eps
        self.K = K

    def forward(self, x):
        N, C, D, H, W = x.size()
        # print(x.shape)
        # print(N,C,D,H,W)
        # print(self.group_num)
        # 将数据重塑为(N,group_num, H, W)的形状
        x = x.view(N, self.group_num * C, H, W)
        # print(x.shape)

        # 计算每个组的均值和标准差
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + self.eps)

        # 将数据重塑回原始形状
        x = x.view(N, C, D, H, W)
        return x * self.gamma + self.beta


class DSP(nn.Module):
    def __init__(self,
                 group_num: int = 50,
                 K=32
                 ):
        super().__init__()
        self.gn = DeepNormal3D(group_num=group_num)
        self.sigomid = nn.Sigmoid()
        self.K = K

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = F.softmax(self.gn.gamma, dim=0)
        # print(w_gamma.shape)
        # 按照权重大小降序排列
        sorted_weights, sorted_indices = torch.sort(w_gamma.view(-1), descending=True)

        selected_slices_indices = sorted_indices[:self.K]
        # print(selected_slices_indices)
        selected_slices = x[:, :, selected_slices_indices, :, :]
        return selected_slices


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x = torch.randn((3, 1, 50, 512, 512)).to(device)
    deep_select = DSP()
    deep_select.to(device)
    out = deep_select(x)
    print(out.shape)
