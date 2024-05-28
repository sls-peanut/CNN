# -*- codeing = utf-8 -*-
# @Software : PyCharm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transforms.RandomHorizontalFlip(p=0.5)---以0.5的概率对图片做水平横向翻转
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# transforms.ToTensor()---shape从(H,W,C)->(C,H,W), 每个像素点从(0-255)映射到(0-1):直接除以255
# transforms.Normalize---先将输入归一化到(0,1),像素点通过"(x-mean)/std",将每个元素分布到(-1,1)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(std=(0.485, 0.456, 0.406), mean=(0.226, 0.224, 0.225))])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train,
                                 download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform,
                                download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 构建 VGGNet16 网络模型
class VGGNet16(nn.Module):
    def __init__(self):
        super(VGGNet16, self).__init__()

        self.Conv1 = nn.Sequential(
            # CIFAR10 数据集是彩色图 - RGB三通道, 所以输入通道为 3, 图片大小为 32*32
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            # inplace-选择是否对上层传下来的tensor进行覆盖运算, 可以有效地节省内存/显存
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层
        self.fc = nn.Sequential(

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # 使一半的神经元不起作用，防止参数量过大导致过拟合
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        # 四个卷积层
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)

        # 数据平坦化处理，为接下来的全连接层做准备
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


# 初始化模型
model = VGGNet16().to(device)

# 构造损失函数和优化器
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=0.001)

# 动态更新学习率------每隔step_size : lr = lr * gamma
schedule = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.6, last_epoch=-1)

loss_list = []


# train
def train(epoch):
    start = time.time()
    for epoch in range(epoch):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):

            inputs, labels = inputs.to(device), labels.to(device)

            # 将数据送入模型训练
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels).to(device)

            # 重置梯度
            opt.zero_grad()
            # 计算梯度，反向传播
            loss.backward()
            # 根据反向传播的梯度值优化更新参数
            opt.step()

            running_loss += loss.item()
            loss_list.append(loss.item())

            # 每一百个 batch 查看一下 loss
            if (i + 1) % 100 == 0:
                print('epoch = %d , batch = %d , loss = %.6f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # 每一轮结束输出一下当前的学习率 lr
        lr_1 = opt.param_groups[0]['lr']
        print("learn_rate:%.15f" % lr_1)
        schedule.step()

    end = time.time()
    # 计算并打印输出你的训练时间
    print("time:{}".format(end - start))

    # 训练过程可视化
    plt.plot(loss_list)
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.savefig('./train_img.png')
    plt.show()


# Test
def verify():
    model.eval()
    correct = 0.0
    total = 0
    # 训练模式不需要反向传播更新梯度
    with torch.no_grad():
        print("=========================test=========================")
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
            total += inputs.size(0)
            correct += torch.eq(pred, labels).sum().item()

    print("Accuracy of the network on the 10000 test images:%.2f %%" % (100 * correct / total))
    print("======================================================")


if __name__ == '__main__':
    train(100)
    verify()
    # VGGNet: 所有卷积层全部使用使用3*3的卷积核, 两个3*3=一个5*5 同时可以减少参数量, 加深神经网络的深度
    # 使用 VGGNet-16 的神经网络训练 CIFAR10 数据集的准确率在 82% 左右



