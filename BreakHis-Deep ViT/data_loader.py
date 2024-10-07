from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# def get_data_loaders(data_dir, batch_size, train=False):
#     if train:
#         transform = transforms.Compose(
#             [
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomVerticalFlip(),
#                 transforms.RandomApply(
#                     [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2
#                 ),
#                 transforms.Resize(256),
#                 transforms.CenterCrop(32),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             ]
#         )
#     else:
#         transform = transforms.Compose(
#             [
#                 transforms.Resize(256),
#                 transforms.CenterCrop(32),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             ]
#         )
#
#     all_data = datasets.ImageFolder(data_dir, transform=transform)
#     print(f'all_data:{all_data}')
#     data_lengths = [int(len(all_data) * x) for x in (0.7, 0.15, 0.15)]  # 导致精度损失,造成最终各数据集长度之和不等于原始数据集长度。
#
#     if train:
#         train_data, val_data, _ = random_split(all_data, data_lengths)
#         train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#         return train_loader
#     else:
#         _, val_data, test_data = random_split(all_data, data_lengths)
#         val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
#         return val_loader, test_loader
def get_data_augmentation(train=False):
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2
                ),
                transforms.Resize(256),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )


def get_data_loaders(data_dir, batch_size, train=False):
    transform = get_data_augmentation(train)
    all_data = datasets.ImageFolder(data_dir, transform=transform)
    print(f'all_data: {all_data}')

    # 计算原始数据集的总长度
    total_length = len(all_data)

    # 定义训练集、验证集和测试集的比例
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # 计算各个数据集的长度
    train_length = int(total_length * train_ratio)
    val_length = int(total_length * val_ratio)
    test_length = total_length - train_length - val_length

    # 确保总长度正确
    assert train_length + val_length + test_length == total_length

    # 进行数据集拆分
    train_data, val_data, test_data = random_split(all_data, [train_length, val_length, test_length])

    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    if train:
        return train_loader
    else:
        return val_loader, test_loader

