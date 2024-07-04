import torch


def cosine_beta_schedule(timesteps, s=0.008):
    """
    :param timesteps: 表示时间步长的数量，即扩散过程中的时间步数
    :param s: 一个小的常数, 默认为 0.008 ,用于调整余弦函数的平移
    原始论文 —— https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1  # 时间步数加一，用于生成 x 的序列长度

    # 使用 torch.linspace 函数生成的线性序列, 从 0 到 timesteps ,长度为 steps
    x = torch.linspace(0, timesteps, steps)

    # 计算余弦函数的平方, 用于生成逐步累积的 alpha 值
    # ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5 将 x 的范围平移并缩放到 (0, π/2) 之间, 然后对其应用余弦函数并平方
    # 归一化 alphas_cumprod, 使其以 1 开始
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # 根据 alpha 值计算 beta 值
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    # 将 betas 值限制在 0.0001 和 0.9999 之间
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    """
    定义了一个线性方差计划, 用于在前向扩散过程中逐步添加噪声.
    :param timesteps: 表示时间步长的数量, 即扩散过程中的时间步数
    :return: 返回一个包含 timesteps 个值的一维张量，这些值从 beta_start 到 beta_end 线性均匀分布。
    """
    beta_start = 0.0001
    beta_end = 0.02

    # torch.linspace 用于生成在指定区间内均匀分布的数值序列。
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    """

    :param timesteps: 表示时间步长的数量, 即扩散过程中的时间步数
    :return: 返回一个包含 timesteps 个值的一维张量，这些值从 beta_start ** 0.5 到 beta_end ** 0.5 线性均匀分布, 然后再逐元素平方
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    """
    定义了一个基于 sigmoid 函数的方差系数调度计划
    :param timesteps: 表示时间步长的数量, 即扩散过程中的时间步数
    :return:
    """
    beta_start = 0.0001
    beta_end = 0.02

    # 使用 torch.linspace 函数生成的线性序列，范围从 -6 到 6
    betas = torch.linspace(-6, 6, timesteps)

    # 将 sigmoid 序列扩展到 beta_start 和 beta_end 之间的范围
    # 将序列的范围平移到 beta_start 和 beta_end 之间
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
