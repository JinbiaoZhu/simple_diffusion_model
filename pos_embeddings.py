import math

import torch
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    这段代码定义了一个 SinusoidalPositionEmbeddings 类, 用于生成正弦位置嵌入.
    这种嵌入方式在 Transformer 网络中常用, 以提供输入数据的位置/时序/噪声水平等信息。
    """

    def __init__(self, dim):
        """
        类初始化.
        :param dim: 嵌入的维度.
        """
        super().__init__()  # 调用父类 nn.Module 的初始化方法.
        self.dim = dim

    def forward(self, time):
        """
        :param time: 输入时间序列张量, 形状为 (batch_size, ).
        :return: 时间相关嵌入.
        """
        device = time.device  # 获取输入张量所在的设备 (如 GPU 或 CPU)
        half_dim = self.dim // 2  # 计算嵌入维度的一半.

        # math.log(10000) / (half_dim - 1): 计算一个常数，用于生成正弦和余弦位置嵌入。
        # torch.arange(half_dim, device=device) * -embeddings: 生成从 0 到 half_dim-1 的数列，并乘以上述常数。
        # torch.exp(...): 对每个元素求指数，得到指数序列。
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # time[:, None]: 将时间序列张量扩展一个维度, 形状变为 (batch_size, 1).
        # embeddings[None, :]: 将指数序列扩展一个维度, 形状变为 (1, half_dim).
        # time[:, None] * embeddings[None, :]: 逐元素相乘且使用广播机制, 生成形状为 (batch_size, half_dim) 的位置嵌入张量。
        embeddings = time[:, None] * embeddings[None, :]

        # embeddings.sin(): 计算位置嵌入张量的正弦值.
        # embeddings.cos(): 计算位置嵌入张量的余弦值.
        # torch.cat(..., dim=-1): 在最后一个维度上拼接正弦和余弦值, 得到形状为 (batch_size, dim) 的嵌入张量.
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


if __name__ == "__main__":
    pos_embedding = SinusoidalPositionEmbeddings(dim=256)
    # 随机生成 (128,) 形状的时序信息用于模拟数据集的时序分布
    times = torch.randint(1, 1024, (128,))
    embeddings = pos_embedding(times)
    print(f"This embedding's shape is {embeddings.shape}")

    # This embedding's shape is torch.Size([128, 256])
