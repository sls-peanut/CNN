import torch
from torchvision import transforms
import datasets
from tqdm import tqdm

"""
这段代码的目的是计算用于图像预处理的均值和标准差。
"""

N_CHANNELS = 3  # 设置输入图像的通道数为 3。

dataset = datasets.BreaKHis('binary', 'train', magnification=None,
                            transform=transforms.ToTensor())  # 加载 BreaKHis 数据集的训练集,并将图像转换为 PyTorch 张量。
full_loader = torch.utils.data.DataLoader(dataset, shuffle=False,
                                          num_workers=8)  # 创建一个数据加载器 full_loader，它会在训练集上进行遍历。shuffle=False 表示不打乱数据集顺序, num_workers=8 表示使用 8 个子进程加载数据。

# 初始化 3 个通道的均值和标准差张量。
mean = torch.zeros(3)
std = torch.zeros(3)
print('==> Computing mean and std..')
'''
使用 tqdm 遍历数据加载器,计算每个通道的均值和标准差:
对于每个输入样本 inputs(shape 为 (batch_size, 3, height, width)),
遍历 3 个通道,累加每个通道的均值和标准差。
最后除以数据集的总样本数,得到最终的均值和标准差。
打印出计算得到的均值和标准差。
这样的预处理步骤是很多深度学习模型训练的标准做法,可以帮助模型更好地收敛。将图像数据标准化到均值为 0,标准差为 1 的分布,可以提高模型的泛化能力。
'''
for inputs, _labels in tqdm(full_loader, total=len(full_loader)):
    for i in range(N_CHANNELS):
        mean[i] += inputs[:, i, :, :].mean()
        std[i] += inputs[:, i, :, :].std()
mean.div_(len(dataset))
std.div_(len(dataset))
print(mean, std)
