import torch
from torch import nn
from einops import rearrange, einsum


class Attention(nn.Module):
    """
    定义了一个 Attention 类, 它实现了自注意力机制.
    """
    def __init__(self, dim, heads=4, dim_head=32):
        """

        :param dim: 输入通道数
        :param heads: 注意力头的数量，默认为 4
        :param dim_head: 每个注意力头的维度, 默认为 32
        """
        super().__init__()
        self.scale = dim_head ** -0.5  # 缩放因子, 用于缩放 query 张量, 防止数值过大
        self.heads = heads  # 注意力头的数量
        hidden_dim = dim_head * heads  # 每个注意力头维度与头数量的乘积

        # 1x1 卷积层, 用于生成查询 (Query) 键 (Key) 和值 (Value) 张量
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # 1x1 卷积层, 用于生成输出张量
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        前向操作
        :param x: 输入张量, 形状为 (batch_size, channels, height, width)
        :return: 注意力分数
        """
        # b=batch_size, c=channels, h=height, w=width
        b, c, h, w = x.shape

        # self.to_qkv(x) 输出结果: (batch_size, hidden_dim * 3, height, width)
        # .chunk() 结果: (batch_size, hidden_dim, height, width)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # 对于每个 query key 和 value 都有: (batch_size, hidden_dim, height, width)
        # 把第一个位置的维度拆分: (batch_size, heads, dim_head, height, width)
        # 进行张量的合并: (batch_size, heads, dim_head, (height*width))
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale  # query 乘以缩放因子

        # 进行矩阵乘法: (batch_size, heads, (height*width), (height*width))
        sim = einsum(q, k, "b h d i, b h d j -> b h i j")

        # 稳定数值，减去每行的最大值
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        # 对注意力得分进行 softmax 操作，得到注意力权重.
        attn = sim.softmax(dim=-1)

        # attn: (batch_size, heads, (height*width), (height*width))
        # v: (batch_size, heads, dim_head, (height*width))
        # out: (batch_size, heads, (height*width), dim_head)
        out = einsum(attn, v, "b h i j, b h d j -> b h i d")

        # out: (batch_size, heads*dim_head=hidden_dim, height, width)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        # return: (batch_size, dim, height, width)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    定义了一个 LinearAttention 类, 它实现了线性自注意力机制.
    """
    def __init__(self, dim, heads=4, dim_head=32):
        """

        :param dim: 输入通道数
        :param heads: 注意力头的数量，默认为 4
        :param dim_head: 每个注意力头的维度, 默认为 32
        """
        super().__init__()
        self.scale = dim_head ** -0.5  # 缩放因子, 用于缩放 query 张量, 防止数值过大
        self.heads = heads  # 注意力头的数量
        hidden_dim = dim_head * heads  # 每个注意力头维度与头数量的乘积

        # 1x1 卷积层, 用于生成查询 (Query) 键 (Key) 和值 (Value) 张量
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # 1x1 卷积层, 用于生成输出张量
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        """
        前向操作
        :param x: 输入张量, 形状为 (batch_size, channels, height, width)
        :return: 注意力分数
        """
        # b=batch_size, c=channels, h=height, w=width
        b, c, h, w = x.shape

        # self.to_qkv(x) 输出结果: (batch_size, hidden_dim * 3, height, width)
        # .chunk() 结果: (batch_size, hidden_dim, height, width)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # 对于每个 query key 和 value 都有: (batch_size, hidden_dim, height, width)
        # 把第一个位置的维度拆分: (batch_size, heads, dim_head, height, width)
        # 进行张量的合并: (batch_size, heads, dim_head, (height*width))
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        # 进行 .softmax 操作不会改变维度
        # q: (batch_size, heads, dim_head, (height*width))
        q = q.softmax(dim=-2)

        # k: (batch_size, heads, dim_head, (height*width))
        # v: (batch_size, heads, dim_head, (height*width))
        k = k.softmax(dim=-1)

        q = q * self.scale  # query 乘以缩放因子

        # context: (batch_size, heads, dim_head, dim_head)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        # q: (batch_size, heads, dim_head, (height*width))
        # out: (batch_size, heads, dim_head, (height*width))
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)

        # out: (batch_size, (heads*dim_head)=hidden_dim, height, width)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)

        # out: (batch_size, dim, height, width)
        return self.to_out(out)
