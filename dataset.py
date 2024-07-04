from datasets import load_dataset, load_from_disk
from torchvision import transforms
from torch.utils.data import DataLoader


# 当文件路径中没有数据集时, 从 HuggingFace Hub 中下载 fashion_mnist 数据集
# 注意这里需要配置节点 (魔法上网)
# dataset = load_dataset("fashion_mnist", trust_remote_code=True)
# dataset.save_to_disk("./fashion_mnist")

# 如果文件中有 fashion_mnist 数据集, 则直接从磁盘中导入
dataset = load_from_disk("./fashion_mnist")
image_size = 28   # 设置图像的大小
channels = 1      # 设置图像的通道, 即单通道
batch_size = 128  # 设置批次大小

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),          # 随机水平翻转
    transforms.ToTensor(),                      # 转成 PyTorch 张量, 并从 [0,255] 放缩至 [0,1]
    transforms.Lambda(lambda t: (t * 2) - 1)    # 将元素值从 [0,1] 水平放缩到 [-1,1]
])


def transforms(examples):
    """
    定义并应用变换函数，将 examples 中的 image 带入 transform 函数做变换, 然后转换为 pixel_values
    :param examples: 数据集中的 "每一条"
    :return: 变换完的数据集
    """
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]
    return examples


# 并删除 label 列
transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

# 创建数据加载容器
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    batch = next(iter(dataloader))
    print(len(batch))  # 1
    print(len(batch["pixel_values"]))  # 128
    print(type(batch["pixel_values"][0]), batch["pixel_values"][0].shape)
    # <class 'torch.Tensor'> torch.Size([1, 28, 28])
