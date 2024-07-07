# -*- coding = utf-8 -*-
# @Time :2023/5/2 18:19
# @Author:sls
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

# x.shape  # 形状    x.size()  # 尺寸    x.ndim  # 维数
# torch.Size([64, 3, 32, 32])
# 64表示训练集batch_size大小，CHW 3是图像通道数Channel，32是图像高度Height，32是图像宽度Wirth，图像尺寸 32*32，维度个数是4
dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # x.shape  # 形状    x.size()  # 尺寸    x.ndim  # 维数
    # torch.Size([64, 3, 32, 32])
    # 64表示训练集batch_size大小，CHW 3是图像通道数Channel，32是图像高度Height，32是图像宽度Wirth，图像尺寸 32*32，维度个数是4
    # print(imgs.size())
    # print(imgs.ndim)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    print(torch.reshape(imgs, (1, 1, 1, -1)))
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)
