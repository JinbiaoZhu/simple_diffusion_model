from inspect import isfunction
from einops.layers.torch import Rearrange
from torch import nn


def exists(x) -> bool:
    """
    返回一个布尔值 (True/False), 用于表示这个变量是否是 None .
    如果 x 不是 None 则返回 True ; x 是 None 则返回 False .
    :param x: 待检测的变量
    :return: 布尔值 (True/False)
    """
    return x is not None


def default(val, d):
    """
    为变量设置默认值.
    如果变量 val 不是 None 的话, 直接返回;
    如果变量 val 是 None 且默认值 d 是数值的话, 直接返回;
    如果变量 val 是 None 且默认值 d 是函数名的话, 返回函数的调用结果.
    :param val: 待设置的变量
    :param d: 默认值 (函数)
    :return: 设置结果
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    """
    用于将 num 数量个元素按每 divisor 为一组进行分组, 剩下的元素额外增添一组.
    :param num: 元素的数量
    :param divisor: 每组的数量
    :return: 一个列表, 列表内的每个元素都表示一组的数量
    """
    groups = num // divisor
    remainder = num % divisor  # 求余数
    arr = [divisor] * groups  # 用每组数量 divisor 作为元素构建列表
    if remainder > 0:  # 如果有剩下的, 额外增加一个元素
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    """
    基于 PyTorch 的 nn.Module 构建的残差网络结构
    注意: 这里面没有具体的模型参数, 不是一个具体的网路.
    """

    def __init__(self, fn):
        """
        类的构造函数
        :param fn: 具体的网络模型!
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """

        :param x: 输入具体的网路模型的张量
        :param args: 剩下可能需要的参数
        :param kwargs: 剩下可能需要的参数
        :return: 具体的神经网络输出 且 结合了残差计算!
        """
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    """
    对图像进行上采样和一次卷积计算.
    :param dim: 图像输入维度
    :param dim_out: 图像输出维度
    :return: 经过上采样和卷积运算后的张量
    """
    return nn.Sequential(  # 创建一个顺序容器，把传入的模块依次加入到容器中执行。在这里，容器中包含两个模块。
        # 这是一个上采样模块，它将输入的图像张量按照指定的比例进行上采样.
        # scale_factor=2 表示将输入的图像沿着宽度和高度方向都放大两倍.
        # mode="nearest" 表示使用最近邻插值法进行像素的填充.
        nn.Upsample(scale_factor=2, mode="nearest"),
        # 卷积网络接受上一步上采样后的输出作为输入, 并对其进行二维卷积操作.
        # 输入通道数 dim, 输出通道数 dim_out (如果没有提供则默认为输入通道数 dim), 卷积核大小为 3x3;
        # padding=1 表示在图像边缘填充一层零像素.
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    """
    输出一个下采样后的张量, 每个通道的分辨率下降但是通道数变多.
    不再使用步幅卷积或池化操作进行下采样.
    :param dim: 输入张量的通道数.
    :param dim_out: 输出张量的通道数。如果未指定，则默认为输入通道数 dim .
    :return: 输出一个下采样后的张量.
    """
    return nn.Sequential(
        # Rearrange 是一个重排张量的操作 (需要 einops 库).
        # 输入张量的形状为 (batch_size, channels, height, width)。
        # 重排操作的含义是将高度和宽度分别拆分为 p1 和 p2 块, 然后将这些块展开到通道维度.
        # 这里 p1=2 和 p2=2 表示将每个 2x2 的块展开.
        # 重排后的形状为 (batch_size, channels * 4, height / 2, width / 2).
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # 输入通道数为 dim * 4 (因为经过重排后通道数变为原来的 4 倍).
        # 输出通道数为 dim_out (如果未指定，则默认为输入通道数 dim).
        # 卷积核大小为 1x1 ,所以不会改变张量的高度和宽度, 只会改变通道数.
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )
