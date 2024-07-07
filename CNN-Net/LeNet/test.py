# -*- coding = utf-8 -*-
# @Time :2023/4/30 21:14
# @Author:sls
# @File :test.py
# @Software:PyCharm
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform_test, download=True)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)
# shuffle是否打乱洗牌    drop_last是否丢弃最后的all/batch_size
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试集中第一张图片
img, target = test_set[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        print(imgs.shape)
        print(targets)
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step += 1

writer.close()

print(f'Epoch {epoch + 1}: Train Loss: {epoch:.4f}')
print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, epoch, epoch / len(test_loader)))
# print(test_set[0])
# print(test_set.classes)

# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# print(test_set[0])
# writer = SummaryWriter("P10test")
# for i in range(10):
#     img, target = test_set[i]
#     writer.add_image("test_set", img, i)
# writer.close()
