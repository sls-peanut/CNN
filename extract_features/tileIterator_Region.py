# -*- coding = utf-8 -*-
# @Time :2024/5/29 14:58
# @Author:sls
# @FIle:tileIterator_Region.py
# @Annotation: 该tileIterator()函数提供了一个迭代器，用于以任何所需的分辨率以逐块的方式依次迭代整个幻灯片或幻灯片内的感兴趣区域 (ROI)。
"""除此之外，以下是 tileIterator 的主要可选参数，涵盖了图像分析的大多数分块迭代用例：
region- 允许您指定幻灯片内的 ROI。
scale- 允许您指定所需的放大倍数/分辨率。
tile_size- 允许您指定图块的大小。
tile_overlap- 允许您指定相邻图块之间的重叠量。
format- 允许您指定图块图像的格式（numpy 数组或 PIL 图像）。
在每次迭代中，tileIterator 都会输出一个包含以下内容的字典：
tile- 裁剪的图块图像，仅当明确访问字典的此元素时才会延迟加载或计算。
format- 瓷砖的格式。
x, y- （左、上）以当前放大像素为单位的坐标。
width, height- 当前图块的当前放大像素大小。
level- 当前图块的级别。
magnification- 放大当前图块。
mm_x, mm_y- 当前图块像素的大小（以毫米为单位）。
gx, gy- （左、上）以基本/最大分辨率像素为单位的坐标。
gwidth, gheight- 当前图块的基本/最大分辨率像素大小。
图块图像的延迟加载使我们能够快速迭代图块并根据图块元数据有选择地处理感兴趣的图块。
下面的代码显示了如何遍历幻灯片中具有特定图块大小和特定分辨率的 ROI"""
import large_image
import numpy as np
from IPython.core.display_functions import display
from matplotlib import pyplot as plt


def tileIterator_svs(wsi_path):
    num_tiles = 0
    tile_means = []
    tile_areas = []
    ts = large_image.getTileSource(wsi_path)
    for tile_info in ts.tileIterator(
            region=dict(left=5000, top=5000, width=20000, height=20000, units='base_pixels'),
            scale=dict(magnification=20),
            tile_size=dict(width=1000, height=1000),
            tile_overlap=dict(x=50, y=50),
            format=large_image.tilesource.TILE_FORMAT_PIL,
    ):

        if num_tiles == 100:
            print('Tile-{} = '.format(num_tiles))
            display(tile_info)

        im_tile = np.array(tile_info['tile'])
        tile_mean_rgb = np.mean(im_tile[:, :, :3], axis=(0, 1))

        tile_means.append(tile_mean_rgb)
        tile_areas.append(tile_info['width'] * tile_info['height'])

        num_tiles += 1

    slide_mean_rgb = np.average(tile_means, axis=0, weights=tile_areas)

    print('Number of tiles = {}'.format(num_tiles))
    print('Slide mean color = {}'.format(slide_mean_rgb))


def getSingleTile_svs(wsi_path):
    """getSingleTile:该getSingleTile()函数可用于直接获取图块迭代器中特定位置的图块。除了上述的参数tileIterator之外，
    它还接受一个 tile_position 参数，可用于指定感兴趣的图块的线性位置。"""
    ts = large_image.getTileSource(wsi_path)
    pos = 1000
    tile_info = ts.getSingleTile(
        tile_size=dict(width=1000, height=1000),
        scale=dict(magnification=20),
        tile_position=pos, )
    plt.imshow(tile_info['tile'])
    plt.show()


def getRegion_svs(wsi_path):
    ts = large_image.getTileSource(wsi_path)
    """该getRegion()函数可用于通过以下两个参数以任意比例 / 放大倍数获取幻灯片内的矩形感兴趣区域(ROI)：region - 包含
    ROI的（左、上、宽、高、单位）的字典
    scale - 包含像素的放大倍数或物理尺寸（mm_x，mm_y）的字典"""
    im_roi, _ = ts.getRegion(
        region=dict(left=10000, top=10000, width=1000, height=1000, units='base_pixels'),
        format=large_image.tilesource.TILE_FORMAT_NUMPY, )
    plt.imshow(im_roi)
    plt.show()
    im_low_res, _ = ts.getRegion(
        scale=dict(magnification=1.25),
        format=large_image.tilesource.TILE_FORMAT_NUMPY)
    plt.imshow(im_low_res)
    plt.show()


def convertRegionScale_svs(wsi_path):
    """该convertRegionScale()函数可用于将区域从一个比例/放大率转换为另一个比例/放大率，如以下示例所示"""
    ts = large_image.getTileSource(wsi_path)
    tr = ts.convertRegionScale(
        sourceRegion=dict(left=5000, top=5000, width=1000, height=1000,
                          units='mag_pixels'),
        sourceScale=dict(magnification=20),
        targetScale=dict(magnification=10),
        targetUnits='mag_pixels',
    )

    display(tr)


def getRegionAtAnotherScale_svs(wsi_path):
    # 以低得多的比例获得以基本分辨率定义的大区域
    ts = large_image.getTileSource(wsi_path)
    im_roi, _ = ts.getRegionAtAnotherScale(
        sourceRegion=dict(left=5000, top=5000, width=10000, height=10000,
                          units='base_pixels'),
        targetScale=dict(magnification=1.25),
        format=large_image.tilesource.TILE_FORMAT_NUMPY)
    print("------------------")
    print(im_roi.shape)


if __name__ == '__main__':
    wsi_path_svs = r'D:\AI\deep Learning\CNN\extract_features\tiff\TCGA-02-0010-01Z-00-DX4.07de2e55-a8fe-40ee-9e98-bcb78050b9f7.svs'
    # tileIterator_svs(wsi_path_svs)
    # getSingleTile_svs(wsi_path_svs)
    # getRegion_svs(wsi_path_svs)
    convertRegionScale_svs(wsi_path_svs)
    getRegionAtAnotherScale_svs(wsi_path_svs)
