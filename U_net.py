from functools import partial

import torch
import torch.nn as nn

from utils_functions import default, Residual, Downsample, Upsample
from resnet import ResnetBlock
from pos_embeddings import SinusoidalPositionEmbeddings
from attention import LinearAttention, Attention
from norm import PreNorm


class Unet(nn.Module):
    def __init__(self, dim,
                 init_dim=None, out_dim=None,
                 dim_mults=(1, 2, 4, 8), channels=3,
                 self_condition=False,
                 resnet_block_groups=4):
        """
        这段代码定义了一个 Unet 类，用于图像处理任务中的 U-Net 模型
        :param dim: 基础通道数
        :param init_dim: 初始化卷积层的通道数, 如果未提供则默认为 dim
        :param out_dim: 输出通道数, 如果未提供则默认为输入图像的通道数
        :param dim_mults: 用于控制每个阶段的通道数的倍数
        :param channels: 输入图像的通道数，默认为 3 (RGB 图像)
        :param self_condition: 是否使用自我条件
        :param resnet_block_groups: ResNet块中组归一化的组数
        """
        super().__init__()

        # 确定维度和初始卷积, 确定是否使用 self_condition
        self.channels = channels
        self.self_condition = self_condition

        # 如果使用 self_condition ,输入通道数变为原来的两倍
        input_channels = channels * (2 if self_condition else 1)

        # 初始化卷积层, 将输入通道数转换为 init_dim , 当 init_dim 没有定义时, 初始化 init_dim=dim
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        # 通过 dim_mults 确定每个阶段的通道数
        # map 是一个内置函数, 用于将一个函数应用到一个可迭代对象的每一个元素上, 并返回一个迭代器.
        # 在这里，它将 lambda m: dim * m 应用于 dim_mults 的每一个元素.
        # 也就是对 dim_mults 内部的所有元素都乘以一个 dim 数值
        # * 运算符在这里用于解包操作, 它将 map 函数返回的迭代器解包成一个列表或元组，具体取决于上下文。
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        # in_out 列表用于定义每一层的输入和输出通道数
        # dims[:-1] 获取列表从第一个元素到倒数第二个元素的子列表
        # dims[1:] 获取列表从第二个元素到最后一个元素的子列表
        # zip 函数将两个或多个可迭代对象 (例如列表) 的元素配对起来，形成一个由元组组成的迭代器
        # list 函数将 zip 函数返回的迭代器转换为一个列表
        in_out = list(zip(dims[:-1], dims[1:]))

        # 使用偏函数简化 ResNet 块的定义
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4  # 时间嵌入的维度

        # 采用正弦位置嵌入, 对时间信息进行编码, 然后再用线性层扩张到时间维度 time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),  # (batch_size, dim)
            nn.Linear(dim, time_dim),           # (batch_size, time_dim)
            nn.GELU(),                          # (batch_size, time_dim)
            nn.Linear(time_dim, time_dim),      # (batch_size, time_dim)
        )

        # self.downs 定义下采样层容器
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            # 每个下采样阶段包含两个 ResNet 块, 一个自注意力模块, 以及一个下采样操作
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),  # 两个 ResNet 块
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),        # 一个自注意力模块
                        Downsample(dim_in, dim_out)                                # 一个下采样操作
                        if not is_last
                        # 如果是最后一层还有一个输出卷积的操作
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # 中间层包含两个 ResNet 块和一个自注意力模块。
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # 使用 ModuleList 存储每个上采样层的模块
        # 注意遍历过程有 reversed() 操作
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            # 每个上采样阶段包含两个 ResNet 块，一个自注意力模块，以及一个上采样操作。
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),  # 两个 ResNet 块
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),                 # 一个自注意力模块
                        Upsample(dim_out, dim_in)                                             # 一个上采样操作
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        # 最终层包含一个 ResNet 块和一个卷积层，将通道数转换为输出通道数
        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):

        # 如果使用 self_condition , 将 self_condition 张量与输入张量拼接
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        # 对输入进行初始卷积并保存用于残差计算
        # 针对 Fashion MNIST 数据集, 输入 x : (128, 1, 28, 28)
        # 经过 self.init_conv 扩充通道后, 输出 x : (128, 28, 28, 28)
        x = self.init_conv(x)
        r = x.clone()
        # 针对 Fashion MNIST 数据集, 计算时间嵌入, 输入 self.time_mlp 的噪声水平张量的维度是 (128,) 输出张量的维度是 (128, 112)
        # 这就意味着每个时间噪声水平都扩展到了 112 维度的高维空间
        t = self.time_mlp(time)

        # 对输入张量进行多次下采样，并存储中间结果。
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # 下采样前的 x (128, 28, 28, 28) 下采样后的 x (128, 112, 7, 7)
        # 经过中间层的处理
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # 中间层前的 x (128, 112, 7, 7) 中间层后的 x (128, 112, 7, 7)
        # 对输入张量进行多次上采样, 并结合下采样过程中的中间结果
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        # 上采样前的 x (128, 112, 7, 7) 上采样后的 x (128, 28, 28, 28)
        # 将最终结果与初始卷积结果拼接, 并经过最终的 ResNet 块和卷积层
        # 残差连接合并后是 (128, 56, 28, 28)
        x = torch.cat((x, r), dim=1)

        # self.final_res_block 前的 x (128, 56, 28, 28) self.final_res_block 后的 (128, 28, 28, 28)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
