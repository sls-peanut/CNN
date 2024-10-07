# -*- coding = utf-8 -*-
# @Time :2023/5/1 20:53
# @Author:sls
# @File :nn_conv2d.py
# @Software:PyCharm
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):

    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()
print(tudui)

writer = SummaryWriter("./logs")

step = 0
for data in dataloader:
    imgs, label = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30]) ->[xxx,3x 30, 30]

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step += 1
