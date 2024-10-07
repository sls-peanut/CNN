import os
import pandas as pd
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

CSV_PATH = './dataset/BreaKHis_v1/mysplit.csv'
DATASET_ROOT = '../data/'
# DATASET_ROOT = '../data/BreaKHis_v1/'
# DATASET_ROOT = 'D:\\AI\\deep Learning\\CNN\\data\\BreaKHis_v1'


def get_info(name):
    """re.match() 函数将文件名与正则表达式进行匹配。如果匹配成功,则提取各个字段的值并存储在一个字典 info 中。"""
    name = name.split('/')[-1]
    info = dict()
    # 定义了一个正则表达式 pattern，用于匹配文件名的格式。这个格式遵循以下规则:
    # [procedure]_[tumor class]_[tumor type]-[year]-[slide id]-[magnification]-[seq].png
    pattern = re.compile(r'([A-Z]+)_([A-Z])_([A-Z]+)-(\d+)-(\d*[a-zA-Z]*)-(\d+)-(\d+)(\.png)')
    match = pattern.match(name)
    if match == None:
        print(f'Error: invalid filename: {name}')
        return None
    info['procedure'] = match.group(1)  # procedure: 手术或实验名称
    info['tumor_class'] = match.group(2)  # tumor class: 肿瘤类别(A, F, PT, TA, DC, LC, MC, PC)
    info['tumor_type'] = match.group(3)  # tumor type: 肿瘤类型
    info['year'] = match.group(4)  # year: 图像采集年份
    info['slide_id'] = match.group(5)  # slide id: 切片 ID
    info['magnification'] = match.group(6)  # magnification: 放大倍率
    info['seq'] = match.group(7)  # seq: 序列号
    return info


class image_data:
    def __init__(self, path, info, label):
        self.path = path
        self.info = info
        self.label = label


class BreaKHis_generate(Dataset):
    BINARY_LABEL_DICT = {'B': 0, 'M': 1}  # 二分类标签映射
    SUBTYPE_LABEL_DICT = {'A': 0, 'F': 1, 'PT': 2, 'TA': 3, 'DC': 4, 'LC': 5, 'MC': 6, 'PC': 7}  # 亚型(多)分类
    MAGNIFICATION_DICT = {'40': 0, '100': 1, '200': 2, '400': 3}  # 放大倍数
    # LABEL_DICT 字典,其键为任务类型,值为对应的标签映射字典。
    LABEL_DICT = {'binary': BINARY_LABEL_DICT, 'subtype': SUBTYPE_LABEL_DICT, 'magnification': MAGNIFICATION_DICT}

    def __init__(self, task_type, group, magnification=None, transform=None, filter=True):
        assert task_type in ['binary', 'subtype',
                             'magnification'], 'task_type must be one of [binary, subtype, magnification]'
        assert group in ['train', 'dev', 'test'], 'group must be one of [train, dev, test]'
        if magnification:
            magnification = str(magnification)
        assert magnification == None or magnification in ['40', '100', '200',
                                                          '400'], 'magnification must be one of [40, 100, 200, 400]'

        self.magnification = magnification
        self.transform = transform
        self.label_dict = self.LABEL_DICT[task_type]
        self.img_list = []

        label_dict = self.LABEL_DICT[task_type]
        # 使用 os.walk() 函数遍历 DATASET_ROOT 目录及其子目录,root 表示当前目录路径, _ 是占位符,表示忽略当前目录下的子目录, files 是当前目录下的所有文件名列表。
        for root, _, files in os.walk(DATASET_ROOT):
            for file in files:
                if file.endswith('.png') == False:
                    continue
                info = get_info(file)
                if info == None:
                    continue
                if filter and info['slide_id'] == '13412':
                    continue
                # 使用图像路径、图像信息字典 info 和 None 参数创建一个 image_data 对象,并将其添加到 self.img_list 列表中。
                self.img_list.append(image_data(os.path.join(root, file), info, None))
        # 按照放大倍数过滤: 如果提供了 magnification 参数,代码会过滤 self.img_list,只保留放大倍数匹配的图像。
        if magnification:
            self.img_list = [img for img in self.img_list if img.info['magnification'] == magnification]

        self.img_list.sort(key=lambda img: img.path)  # 根据路径排序
        # 7:2:1划分训练测试验证
        if group == 'train':
            self.img_list = self.img_list[3::10] + self.img_list[4::10] + self.img_list[5::10] + \
                            self.img_list[6::10] + self.img_list[7::10] + self.img_list[8::10] + self.img_list[9::10]
        elif group == 'dev':
            self.img_list = self.img_list[1::10] + self.img_list[2::10]
        else:
            self.img_list = self.img_list[0::10]

        img_label_property_dict = {'binary': 'tumor_class', 'subtype': 'tumor_type', 'magnification': 'magnification'}
        for img in self.img_list:
            img.label = label_dict[img.info[img_label_property_dict[task_type]]]

        img_cnt, class_cnt = self.statistics()
        print(
            f'loaded dataset with {img_cnt} images, task_type: {task_type}, group: {group}, magnification: {magnification}')
        print(class_cnt)

    def __getitem__(self, index):
        path = self.img_list[index].path
        label = self.img_list[index].label

        img = Image.open(path)

        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)

        return img, label

    def __len__(self):
        return len(self.img_list)

    def statistics(self):
        class_cnt = {k: 0 for k in self.label_dict.keys()}
        label_invert_dict = {v: k for k, v in self.label_dict.items()}
        for img in self.img_list:
            class_cnt[label_invert_dict[img.label]] += 1
        return len(self.img_list), class_cnt


class BreaKHis(Dataset):
    BINARY_LABEL_DICT = {'B': 0, 'M': 1}
    SUBTYPE_LABEL_DICT = {'A': 0, 'F': 1, 'PT': 2, 'TA': 3, 'DC': 4, 'LC': 5, 'MC': 6, 'PC': 7}
    MAGNIFICATION_DICT = {'40': 0, '100': 1, '200': 2, '400': 3}
    LABEL_DICT = {'binary': BINARY_LABEL_DICT, 'subtype': SUBTYPE_LABEL_DICT, 'magnification': MAGNIFICATION_DICT}

    def __init__(self, task_type, group, magnification=None, transform=None, split_csv=CSV_PATH):
        assert task_type in ['binary', 'subtype',
                             'magnification'], 'task_type must be one of [binary, subtype, magnification]'
        assert group in ['train', 'dev', 'test'], 'group must be one of [train, dev, test]'
        if magnification:
            magnification = str(magnification)
        assert magnification == None or magnification in ['40', '100', '200',
                                                          '400'], 'magnification must be one of [40, 100, 200, 400]'

        self.magnification = magnification
        self.transform = transform
        self.label_dict = self.LABEL_DICT[task_type]
        self.img_list = []

        label_dict = self.LABEL_DICT[task_type]
        df = pd.read_csv(split_csv)

        if magnification:
            df = df[df['mag_grp'] == int(magnification)]
        df = df[df['grp'] == group]
        img_label_property_dict = {'binary': 'tumor_class', 'subtype': 'tumor_type', 'magnification': 'magnification'}
        for _, row in df.iterrows():
            path = os.path.join(DATASET_ROOT, row['path'])
            info = get_info(row['path'].split('/')[-1])
            label = label_dict[info[img_label_property_dict[task_type]]]
            self.img_list.append(image_data(path, info, label))

    def __getitem__(self, index):
        path = self.img_list[index].path
        label = self.img_list[index].label

        img = Image.open(path)

        if self.transform:
            img = self.transform(img)
        else:
            img = ToTensor()(img)

        return img, label

    def __len__(self):
        return len(self.img_list)

    def statistics(self):
        class_cnt = {k: 0 for k in self.label_dict.keys()}
        label_invert_dict = {v: k for k, v in self.label_dict.items()}
        for img in self.img_list:
            class_cnt[label_invert_dict[img.label]] += 1
        return self.magnification, len(self.img_list), class_cnt  # 数据集中图像的放大倍数 图像的总数  class_cnt:每个类别的样本数量


num_classes_dict = {
    'binary': 2,
    'subtype': 8,
    'magnification': 4
}

if __name__ == '__main__':
    for mag in [None, 40, 100, 200, 400]:
        BreaKHis_generate('binary', 'train', mag)
        BreaKHis_generate('binary', 'dev', mag)
        BreaKHis_generate('binary', 'test', mag)
        BreaKHis_generate('subtype', 'train', mag)
        BreaKHis_generate('subtype', 'dev', mag)
        BreaKHis_generate('subtype', 'test', mag)

        print(BreaKHis_generate('binary', 'train', mag).img_list[1].path)

        # train = BreaKHis('subtype', 'train', mag).statistics()[2]
        # test = BreaKHis('subtype', 'test', mag).statistics()[2]
        # for k in train.keys():
        #     print(f'{k}: {train[k] / (train[k] + test[k])}')  # 打印出每个类别在训练集中的样本占比。
