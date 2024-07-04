import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from diffusion import sample
import dataset

# ###################### Training Parameters ######################
dataloader = dataset.dataloader  # 加载数据集容器
image_size = dataset.image_size  # 统一设置图像大小
channels = dataset.channels  # 统一图像通道, 也就是单通道
epochs = 10  # 训练轮数
timesteps = 300  # 经过扩散的时间步

# ###################### Model Testing ######################
# 从本地加载模型进行推理
new_model = torch.load("diffusion_model_trained")
new_model.eval()

# 使用训练好的扩散模型进行推理采样
samples = sample(new_model, image_size=image_size, batch_size=64, channels=channels)

# 展示一组 batch_size 中的第 random_index 张图片的生成结果
random_index = 5
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
plt.savefig("simple.png")

# 展示一组 batch_size 中的第 random_index 张图片的生成结果
random_index = 53

fig = plt.figure()
ims = []
for i in range(timesteps):
    # 把每一个 timestep 的展示图片缓存起来
    im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
    ims.append([im])

# 将缓存起来的图片做成 gif 并保存
animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save('diffusion.gif')
plt.show()
