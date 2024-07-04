from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce, rearrange

from utils_functions import exists


class WeightStandardizedConv2d(nn.Conv2d):
    """
    WeightStandardizedConv2d 类在卷积操作前对权重进行标准化.
    具体来说, 它先计算权重的均值和方差, 然后使用这些统计量对权重进行标准化处理.
    这有助于提高训练的稳定性和效果. 标准化后的权重再用于标准的二维卷积操作.
    权重标准化和组归一化 (Group Normalization) 共同使用效果更佳.
    —— 原始论文链接: https://arxiv.org/abs/1903.10520 .
    """
    def forward(self, x):
        """
        继承 nn.Conv2d 类并改写 .forward 函数.
        :param x: 输入张量.
        :return: 使用标准化后的权重 normalized_weight 进行卷积操作.
        其他参数如 self.bias 偏置, self.stride 步幅, self.padding 填充, self.dilation 扩展和 self.groups 组数,
        都与 nn.Conv2d 保持一致.
        """
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3  # 防止除零的小常数, 根据输入数据类型设置不同的值
        weight = self.weight  # 获取卷积层的权重

        # 计算权重的均值
        # "o ... -> o 1 1 1" 表示将输出的 o 所在的维度的均值计算并保留, 其他维度压缩成 1
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")

        # 计算权重的方差, unbiased=False 表示使用有偏估计
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))

        # 对权重进行标准化, 减去均值, 除以方差加上一个小常数的平方根
        normalized_weight = (weight - mean) / (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        """
        类初始化.
        :param dim: 输入通道数
        :param dim_out: 输出通道数
        :param groups: 用于组归一化的组数，默认值为 8
        """
        super().__init__()
        # 使用 WeightStandardizedConv2d 进行 3x3 卷积, 输入通道数为 dim, 输出通道数为 dim_out
        # 在卷积过程中进行权重标准化, 填充为 1
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)

        # 使用 nn.GroupNorm 进行组归一化, 组数为 groups , 归一化维度为 dim_out
        self.norm = nn.GroupNorm(groups, dim_out)

        # 使用 nn.SiLU 作为激活函数
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        """

        :param x: 输入张量
        :param scale_shift: 一个可选的缩放和偏移参数，默认为 None
        :return:
        """
        # 输入张量 x 首先通过卷积层 self.proj , 然后通过组归一化层 self.norm .
        # 对于 self.proj 在卷积过程中先进行权重标准化再前向计算
        x = self.proj(x)
        x = self.norm(x)

        # 检查 scale_shift 是否存在 (exists 函数在 utils_functions.py 中).
        # 如果 scale_shift 存在, 解包为 scale 和 shift.
        # 对归一化后的张量 x 进行缩放 (scale + 1) 和偏移 (shift) 操作.
        # x = x * (scale + 1) + shift 可以理解为 x = x * scale + shift + x 也就是进行了残差计算
        # 因为正弦位置编码会存在 scale 数值特别小, 增加训练难度, 因此这里增加残差链接
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)  # 对张量 x 应用 SiLU 激活函数
        return x


class ResnetBlock(nn.Module):
    """
    定义一个 ResnetBlock 类, 继承自 nn.Module ,实现了一个具有时间嵌入功能的残差块.
    —— 原始论文链接: https://arxiv.org/abs/1512.03385 .
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        """

        :param dim: 输入的通道数.
        :param dim_out: 输出的通道数.
        :param time_emb_dim: 正弦位置编码的维度.
        :param groups: 用于组归一化的组数，默认值为 8
        """
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )  # 如果提供了 time_emb_dim ,则定义一个包含 SiLU 激活函数和线性层的顺序容器, 否则为 None .

        self.block1 = Block(dim, dim_out, groups=groups)  # 第一个卷积块
        self.block2 = Block(dim_out, dim_out, groups=groups)  # 第二个卷积块
        # 用于匹配输入输出通道数的卷积层. 如果输入和输出通道数相同, 则使用 nn.Identity() 保持输入不变.
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """
        前向计算.
        :param x: 输入张量.
        :param time_emb: 时间嵌入张量.
        :return:
        """
        # 如果 self.mlp 和 time_emb 都存在, 先通过 self.mlp 处理 time_emb,
        # 然后重排列张量, 最后将其分成两个张量 scale 和 shift .
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            # 针对 Fashion MNIST 数据集, 正弦位置编码的维度是 (128, 112) 经过 self.mlp 计算后降维到 (128, 56)
            time_emb = self.mlp(time_emb)
            # 针对 Fashion MNIST 数据集, 将降维到 (128, 56) 的张量扩充维度到 (128, 56, 1, 1)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            # scale_shift 在第 1 维度上拆分成 2 个并返回元组, 每个元组都是 (128, 28, 1, 1) 张量
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)  # 通过第一个卷积块处理输入张量 x ,并应用 scale_shift (如果存在)
        h = self.block2(h)  # 通过第二个卷积块处理 h
        return h + self.res_conv(x)  # 将块的输出 h 与通过 self.res_conv 处理的输入 x 相加，形成残差连接
