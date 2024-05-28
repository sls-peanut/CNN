# -*- coding = utf-8 -*-
# @Time :2024/5/27 15:18
# @Author:sls

import os
import large_image
import numpy as np
import matplotlib.pyplot as plt

# Some nice default configuration for plots
plt.rcParams['figure.figsize'] = 10, 10
plt.rcParams['image.cmap'] = 'gray'


def getMetadata_SVS(wsi_path):
    ts = large_image.getTileSource(wsi_path)
    print('元数据', ts.getMetadata())  # magnification:放大倍数 levels：级别
    print('最大级别的放大倍数', ts.getNativeMagnification())
    print('指定4级别的放大倍数', ts.getMagnificationForLevel(level=4))  # 获取指定级别的放大倍数
    print('当magnification为5.0(level=4)时,每个像素对应的物理尺寸为0.002006毫米(2.006微米)。这意味着图像被放大了5倍,相对于原始分辨率缩小了4倍(scale为4.0)')
    for i in range(ts.levels):
        print('Level级别-{} : {}'.format(i, ts.getMagnificationForLevel(level=i)))
        # get level whose magnification is closest to 10x
    print('magnificant(放大倍数)最接近 10x 的level级别是 = {}'.format(ts.getLevelForMagnification(10)))
    # get level whose pixel width is closest to 0.0005 mm
    print(f'mm_x像素宽度最接近 {0.0005}mm的 magnificant放大级别是 = {ts.getLevelForMagnification(mm_x=0.0005)}')


# if __name__ == '__main__':
#     wsi_path_svs = r'D:\AI\deep Learning\CNN\extract_features\tiff\TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs'
#     getMetadata_SVS(wsi_path_svs)
import numpy as np
from scipy import signal
# 输入图像（RGB三个通道）
X_r = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
X_g = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
X_b = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])
# 卷积核（RGB三个通道）
W_r = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
W_g = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
W_b = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# 对每个通道进行卷积操作
Y_r = signal.convolve2d(X_r, W_r, mode='same') # mode设置为'valid'(默认值),只输出一个值
Y_g = signal.convolve2d(X_g, W_g, mode='same')
Y_b = signal.convolve2d(X_b, W_b, mode='same')

# 将三个特征图在深度维度上进行拼接
Y = np.stack([Y_r, Y_g, Y_b], axis=-1)
print(Y_r)
print("---------------")
print(Y)