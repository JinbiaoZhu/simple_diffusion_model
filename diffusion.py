from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

from schedule import linear_beta_schedule

# ################# Global Parameters #################
timesteps = 300  # 时间步
image_size = 128  # 设置图像为正方形像素
betas = linear_beta_schedule(timesteps=timesteps)  # 使用线性系数调度器
print(f"The first 5 and last 5 betas are: {betas[:5]} and {betas[-5:]}.")

# alphas 通过 1. - betas 计算得到。
# alphas_cumprod 通过 torch.cumprod 计算累积乘积。
# alphas_cumprod_prev 通过填充 alphas_cumprod 的第一个元素且舍弃最后一个元素, 得到前一时间步的累积乘积
# 也就是在第 0 时间步的累计乘积约定是 1 .
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

# sqrt_recip_alphas 计算 alphas 的倒数平方根。
# sqrt_alphas_cumprod 计算 alphas_cumprod 的平方根，用于扩散过程中 q(x_t | x_{t-1}) 的计算。
# sqrt_one_minus_alphas_cumprod 计算 1. - alphas_cumprod 的平方根。
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# posterior_variance 计算后验分布 q(x_{t-1} | x_t, x_0) 的方差。
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# 设置随机数种子
torch.manual_seed(0)


# ################# Functions #################
def extract(a, t, x_shape):
    """
    用于从一维张量 a 中提取对应于时间步长 t 的值, 并将其调整为与输入形状 x_shape 相匹配的形状
    gather 函数用于根据索引 t 从张量 a 中提取值
    提取的值被重新调整形状以匹配输入的 x_shape，并转换回 t 的设备。
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_start, t, noise=None):
    """
    前向扩散过程 —— 给张量乘以相应的系数并增加噪声.
    :param x_start: 被加入噪声的张量
    :param t: 增加噪声的水平
    :param noise: 等待增加的噪声
    :return: 增加噪声后的张量, 对应于 x_start_{t+1}
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    # 抽取 $$\sqrt{\bar{\alpha}_{t}}$$ 和 $$\sqrt{1-\bar{\alpha}_{t}}$$ 系数
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    # 系数, x_start 张量和噪声张量相加
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x_start, t):
    # 给张量添加噪声
    x_noisy = q_sample(x_start, t=t)
    # 将添加噪声后的张量返回成可见图片
    noisy_image = reverse_transform(x_noisy.squeeze())
    return noisy_image


def plot(imgs, orig_image, with_orig=False, row_title=None, **imshow_kwargs):
    # 确保图像列表是一个二维列表, 即使只有一行图像
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    num_rows = len(imgs)  # 获取行数
    num_cols = len(imgs[0]) + with_orig  # 获取列数，如果显示原始图像则列数加一

    # 根据行数和列数实例化绘图句柄
    fig, axs = plt.subplots(figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False)

    for row_idx, row in enumerate(imgs):

        # 如果显示原始图像, 将其添加到每行的开头
        # 注意这里的 row 是列表格式数据, 因此可以和 [image] 相加
        row = [orig_image] + row if with_orig else row

        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]  # 定位待绘制子图的位置
            ax.imshow(np.asarray(img), **imshow_kwargs)  # 显示图像
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])  # 隐藏坐标轴刻度

    if with_orig:
        axs[0, 0].set(title='Original image')  # 设置原始图像的标题
        axs[0, 0].title.set_size(8)  # 设置标题大小

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])  # 为每行设置标题

    plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
    plt.show()  # 显示图像


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    """
    接收一个去噪模型, 输入图像, 时间步和噪声, 生成噪声图像并通过模型预测噪声, 然后根据指定的损失类型计算损失
    :param denoise_model: 去噪模型, 用于预测噪声
    :param x_start: 原始输入图像
    :param t: 时间步, 用于表示当前的时间步骤
    :param noise: 噪声, 如果未提供, 将生成与 x_start 相同形状的随机噪声
    :param loss_type: 损失类型, 可以是 "l1" (L1损失), "l2" (L2损失) 或 "huber" (Huber损失)。
    :return:
    """
    # 如果未提供噪声, 则生成与 x_start 相同形状的随机噪声
    if noise is None:
        noise = torch.randn_like(x_start)

    # 通过函数 q_sample 将原始图像 x_start 加入噪声 noise, 得到噪声图像 x_noisy
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    # 将噪声图像 x_noisy 和时间步 t 输入去噪模型, 得到预测的噪声 predicted_noise
    predicted_noise = denoise_model(x_noisy, t)

    # 根据指定的损失类型计算预测噪声与实际噪声之间的损失
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


transform = Compose([
    Resize(image_size),  # 将输入图像调整为 128x128 像素. 这一步确保所有图像的大小一致.
    CenterCrop(image_size),  # 对图像进行中心裁剪，裁剪后的大小为 128x128 像素
    ToTensor(),  # 将图像转换为 PyTorch 张量, 形状为 (Channel, Height, Width)，并将像素值除以 255 归一化到 [0,1] 范围内
    Lambda(lambda t: (t * 2) - 1),  # 将张量值从 [0,1] 范围调整到 [-1,1] 范围
])

reverse_transform = Compose([
    Lambda(lambda t: (t + 1) / 2),  # 将张量值从 [-1,1] 范围调整到 [0,1] 范围
    Lambda(lambda t: t.permute(1, 2, 0)),  # (Channel, Height, Width) 到 (Height, Width, Channel)
    Lambda(lambda t: t * 255.),  # 图像中的每个像素都变成 [0,255] 范围
    Lambda(lambda t: t.numpy().astype(np.uint8)),  # 转换数据格式
    ToPILImage(),  # 转换到 PIL 可显示的图像
])


@torch.no_grad()
def p_sample(model, x, t, t_index):
    """
    定义了一个函数 p_sample, 用于在扩散模型的反向扩散过程中, 从当前时刻的图像 x 生成下一个时刻的图像
    :param model: 扩散模型, 输入噪声图像和时间步 t, 输出噪声预测
    :param x: 当前时刻的图像张量
    :param t: 当前时间步的张量, 也就是时间正弦位置编码
    :param t_index: 当前时间步的索引
    :return: 当前时刻的图像 x 生成下一个时刻的图像
    """
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # 根据采样公式计算出 x 的前一时刻的图像
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    # 如果是最后一个时间步 t_index == 0 , 直接返回均值 model_mean 作为生成的图像
    # 否则，根据算法 2 的第 4 行, 添加噪声项来生成下一个时刻的图像。
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape):
    """

    :param model: 训练好的模型
    :param shape: 模型的批次形状
    :return: 逐步扩散得到的图片列表
    """
    device = next(model.parameters()).device  # 获取模型存储的设备信息

    b = shape[0]  # 获取 batch_size 信息
    img = torch.randn(shape, device=device)  # 从高斯白噪声开始
    imgs = []  # 设置列表存储信息

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img,
                       torch.full((b,), i, device=device, dtype=torch.long),
                       i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
