from PIL import Image
import matplotlib.pyplot as plt
import torch

from diffusion import transform, get_noisy_image, plot

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open("test_image.png")

# 显示图片且关闭坐标轴
plt.imshow(image)
plt.axis('off')
plt.show()

# 将下载的图像进行变换, 并在第一个维度上扩充
x_start = transform(image).unsqueeze(0)
print(f"x_start's shape is {x_start.shape}.")

# 设置增加噪声的水平为 40 并增加噪声且查看
t = torch.tensor([40])
pic = get_noisy_image(x_start, t)

# 显示图片且关闭坐标轴
plt.imshow(pic)
plt.axis('off')
plt.show()

# 显示一系列的照片
plot([get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]],
     orig_image=image,
     with_orig=True)
