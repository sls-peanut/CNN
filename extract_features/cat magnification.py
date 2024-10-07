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
    print('元数据\n', ts.getMetadata())  # magnification:放大倍数 levels：级别
    print('最大级别的放大倍数:', ts.getNativeMagnification())
    print('指定4级别的放大倍数:', ts.getMagnificationForLevel(level=4))  # 获取指定级别的放大倍数
    print('当magnification为5.0(level=4)时,每个像素对应的物理尺寸为0.002006毫米(2.006微米)。这意味着图像被放大了5倍,相对于原始分辨率缩小了4倍(scale为4.0)')
    for i in range(ts.levels):
        print('Level-{} : {}'.format(i, ts.getMagnificationForLevel(level=i)))
        # get level whose magnification is closest to 10x
    print('magnificant(放大倍数)最接近 10x 的level级别是 = {}'.format(ts.getLevelForMagnification(10)))
    # get level whose pixel width is closest to 0.0005 mm
    print(f'mm_x像素宽度最接近 {0.0005}mm的 magnificant放大级别是 = {ts.getLevelForMagnification(mm_x=0.0005)}')


if __name__ == '__main__':
    wsi_path_svs = r'D:\AI\deep Learning\CNN\extract_features\tiff\TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs'
    # wsi_path_svs = r'D:\AI\deep Learning\CNN\extract_features\tiff\tumor_002.tif'
    getMetadata_SVS(wsi_path_svs)
