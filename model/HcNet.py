# -- coding: utf-8 --
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
import torch.nn.functional as F
from model.Deep_Slice_Prioritizer import DSP
from dataset.dataset import Dataset
import parameter as paras
import os


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VitBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock_3d(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm3d, eps=1e-6), drop_block=None, drop_path=None):
        super().__init__()
        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv3d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv3d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv3d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv3d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """
    CNN feature maps -> Transformer patch embeddings
    将卷积分支上的x_2 channel变为768 ，D,H,W降采样4倍 展平变token 和vit的cls_token 拼接
    """

    def __init__(self, in_c, out_c, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.dw_stride = dw_stride
        # 通道升维
        self.conv_project = nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        # D,H,W 降采样4倍
        self.sample_pooling = nn.AvgPool3d(kernel_size=dw_stride, stride=dw_stride)
        self.ln = norm_layer(out_c)
        self.act = act_layer()

    def forward(self, x, x_t):
        # 特征都升通道升维
        x = self.conv_project(x)
        # 特种图 D,H,W降采样
        x = self.sample_pooling(x)
        # 展平
        x = torch.flatten(x, start_dim=2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)
        # 和vit的cls_token 拼接
        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """
    Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, in_c, out_c, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm3d, eps=1e-6)):
        super().__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(out_c)
        self.act = act_layer()

    def forward(self, x, D, H, W):
        B, _, C = x.shape
        # 去cls_token 转换为特征图
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, D, H, W)
        # 卷积完成后上采样D,H,W 4倍
        x_r = self.act(self.bn(self.conv_project(x_r)))
        x_interpolate = F.interpolate(x_r, size=(D * self.up_stride, H * self.up_stride, W * self.up_stride))

        return x_interpolate


class ConvTransBlock(nn.Module):
    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):
        super().__init__()
        expansion = 4
        self.num_med_block = num_med_block

        self.cnn_block = ConvBlock_3d(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                      groups=groups)
        self.trans_block = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.squeeze_block = FCUDown(in_c=outplanes // expansion, out_c=embed_dim, dw_stride=dw_stride)
        self.expand_block = FCUUp(in_c=embed_dim, out_c=outplanes // expansion, up_stride=dw_stride)

        if last_fusion:
            self.fusion_block = ConvBlock_3d(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True,
                                             groups=groups)
        else:
            self.fusion_block = ConvBlock_3d(inplanes=outplanes, outplanes=outplanes, groups=groups)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        # 1. 进入卷积分支 返回x,x2(该分支中第二个卷积后的输出，channel 是x的1/4)
        # x.shape:[16, 256, 8, 64, 64]  x2.shape:[16, 64, 8, 64, 64] x_t.sahpe [16, 513, 768]
        x, x2 = self.cnn_block(x)

        N, C, D, H, W = x2.shape

        # 2. x_t是之前的vit的输出  x2是卷积分支上输出的特征图
        # 这一步就先将x2变成token 然后拼上x_t的cls_token 然后喂入vit中
        x_st = self.squeeze_block(x2, x_t)
        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        # 3.token 转为特征图
        # 这一将vit 的token(不含cls_token) 通道降维 然后 上采样
        x_t_r = self.expand_block(x_t, D // self.dw_stride, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


class HcNet(nn.Module):
    def __init__(self, patch_size=16, in_chans=1, num_classes=10, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., K=32, group_num=50):
        super().__init__()
        # 分类数目
        self.num_classes = num_classes
        # vit 中 token的维度
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0
        # 生成cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 列表迭代生成没层的drop_path_ratio
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # vit 的分类头
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # cnn 分类头
        self.pooling = nn.AdaptiveAvgPool3d(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

        self.dsp = DSP(K=32, group_num=group_num)
        #  STAGE:0  降维卷积
        self.conv1 = nn.Conv3d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # STAGE:1 (1 conv_trans)双分支但没不包含特征交互
        stage_1_channel = int(base_channel * channel_ratio)  # 64 * 4 = 256
        trans_dw_stride = patch_size // 4  # 4
        self.conv_1 = ConvBlock_3d(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv3d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=self.trans_dpr[0],
                                )

        # STAGE:2 (2,3,4 conv_trans) 堆叠三层conv_trans结构
        init_stage = 2
        fin_stage = depth // 3 + 1  # 5
        for i in range(init_stage, fin_stage):
            self.add_module(f"conv_trans_{i}",
                            ConvTransBlock(inplanes=stage_1_channel, outplanes=stage_1_channel, res_conv=False,
                                           stride=1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                                           num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                           qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                           drop_path_rate=self.trans_dpr[i - 1],
                                           num_med_block=num_med_block
                                           )
                            )

        # STAGE:3 (5,6,7,8 conv_trans) 第一个conv_trans stride=2,channel和上一层的STAGE输出保持一直
        stage_2_channel = int(base_channel * channel_ratio * 2)
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module(f"conv_trans_{i}",
                            ConvTransBlock(in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                                           embed_dim=embed_dim,
                                           num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                           qk_scale=qk_scale,
                                           drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                           drop_path_rate=self.trans_dpr[i - 1],
                                           num_med_block=num_med_block))

        # STAGE:3 (9,10,11,12)
        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block, last_fusion=last_fusion
                            )
                            )
        self.fin_stage = fin_stage
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.dsp(x)
        #  x.shape: [16, 1, 32, 256, 256]
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # STAGE:0  [16, 64, 8, 64, 64]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # SRAGE:1
        # [16, 768, 2, 16, 16] D,H,W 缩小四倍---展平 [16, 512, 768]--- 拼接cls_token [16, 513, 768]
        x = self.conv_1(x_base, return_x_2=False)
        x_t = self.trans_patch_conv(x_base)
        x_t = x_t.flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)
        """
        x： [N, 256, 8, 64, 64]
        x_t = [16, 513, 768] = [N,8*64*64 / (4*4*4) ,768]
        """

        # STAGE:2
        # 2 ~ final
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)
            # print(self.named_modules())
            # print(x.shape, x_t.shape)

        # conv classification
        x_p = self.pooling(x).flatten(1)
        conv_cls = self.conv_cls_head(x_p)

        # trans classification
        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0])
        return [conv_cls, tran_cls]


if __name__ == '__main__':
    nii_dir = r"/remote-home/hongzhangxin/pytorch_project/My_Data/TB_data/TB_last/train"
    train_dataset = Dataset(nii_dir=nii_dir, transform=None)
    x, label,name = train_dataset.__getitem__(index=50)
    x = x.unsqueeze(0)
    model = HcNet(num_classes=2,K=16,group_num=50)
    out = model(x)
    print(out[0].shape, out[1].shape)

