import torch.nn as nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        """

        :param dim: 输入张量的通道数
        :param fn: 要在归一化之后执行的函数
        """
        super().__init__()
        self.fn = fn  # 将传入的函数 fn 保存为实例变量
        # 使用 nn.GroupNorm 进行归一化, 组数为 1 (这相当于对每个通道单独进行归一化)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        """

        :param x: 输入张量
        :return: 返回经过组归一化和特定函数的结果
        """
        x = self.norm(x)  # 输入张量 x 进行归一化操作
        return self.fn(x)  # 将归一化后的张量传递给函数 fn 并返回结果
