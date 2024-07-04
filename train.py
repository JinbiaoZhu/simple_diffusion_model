from pathlib import Path

import torch
import tqdm
from torch.optim import Adam
from torchvision.utils import save_image

from utils_functions import num_to_groups
from U_net import Unet
import dataset
from diffusion import p_losses, sample

# ###################### Training Parameters ######################
results_folder = Path("./results")  # 权重保存路径
results_folder.mkdir(exist_ok=True)  # 如果不存在路径自动生成一个文件夹
save_and_sample_every = 100  # 每 100 个 steps 就推理扩散模型一次, 并存储照片
device = "cuda" if torch.cuda.is_available() else "cpu"  # 显存设置
dataloader = dataset.dataloader  # 加载数据集容器
image_size = dataset.image_size  # 统一设置图像大小
channels = dataset.channels  # 统一图像通道, 也就是单通道
epochs = 10  # 训练轮数
timesteps = 300  # 经过扩散的时间步

# ###################### Training Components ######################
# 初始化模型并放置在显存设备
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
print(f"Model description:")
print(model)
model.to(device)

# 初始化优化器
optimizer = Adam(model.parameters(), lr=1e-3)

# ###################### Training Loop ######################
for epoch in range(epochs):
    print(f"////////////////////////////")
    print(f"This is the {epoch}th epoch.")
    print(f"////////////////////////////")
    for step, batch in tqdm.tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        batch_size = batch["pixel_values"].shape[0]  # batch_size: 128
        batch = batch["pixel_values"].to(device)  # batch: (128, 1, 28, 28)

        # 从最大时间步 timesteps 中均匀采样 (batch_size,) 的时间索引
        # t: (128,)
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = p_losses(model, batch, t, loss_type="huber")

        # 每训练 100*batch_size 张图像就显示一次损失
        if step % 100 == 0:
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()

        # 推理并保存扩散模型推理结果
        # if step != 0 and step % save_and_sample_every == 0:
        #     milestone = step // save_and_sample_every
        #     batches = num_to_groups(4, batch_size)
        #     # 进行模型推理, 采样不同逆扩散过程下的图片
        #     all_images_list = list(map(lambda n:
        #                                sample(model, image_size=image_size, batch_size=n, channels=channels),
        #                                batches))
        #     # all_images = torch.cat(all_images_list, dim=0)
        #     all_images = torch.tensor(all_images_list)
        #     all_images = (all_images + 1) * 0.5
        #     save_image(all_images, str(results_folder / f'train-sample-{milestone}.png'), nrow=6)

# 保存模型权重
torch.save(model, "diffusion_model_trained")
