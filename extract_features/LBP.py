# -*- coding = utf-8 -*-
# @Time :2024/5/23 23:29
# @Author:sls
import cv2
from matplotlib import pyplot as plt
from skimage import io
from skimage.exposure import exposure
from skimage.feature import local_binary_pattern, SIFT, hog, ORB

image10 = cv2.imread('./data/boy.jpeg')  # 读取图像
image11 = cv2.cvtColor(image10, cv2.COLOR_BGR2RGB)  # 按照RGB顺序展示原图
image12 = cv2.cvtColor(image11, cv2.COLOR_BGR2GRAY)  # 灰度转换

image20 = cv2.imread('./data/cars1.jpg')  # 读取图像
image21 = cv2.cvtColor(image20, cv2.COLOR_BGR2RGB)  # 按照RGB顺序展示原图
image22 = cv2.cvtColor(image21, cv2.COLOR_BGR2GRAY)  # 灰度转换

image30 = cv2.imread('./data/zhuan.jpg')  # 读取图像
image31 = cv2.cvtColor(image30, cv2.COLOR_BGR2RGB)  # 按照RGB顺序展示原图
image32 = cv2.cvtColor(image31, cv2.COLOR_BGR2GRAY)  # 灰度转换

image40 = cv2.imread('./data/Pathological.png')  # 读取图像
image41 = cv2.cvtColor(image40, cv2.COLOR_BGR2RGB)  # 按照RGB顺序展示原图
image42 = cv2.cvtColor(image41, cv2.COLOR_BGR2GRAY)  # 灰度转换


def fun_LBP(image):
    # LBP处理
    radius = 1  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数
    lbp = local_binary_pattern(image, n_points, radius)
    return lbp


def fun_HOG(image):
    # HOG

    # 调整图像对比度
    image = exposure.rescale_intensity(image, in_range=(10, 90))

    # 计算HOG描述符
    normalised_blocks, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8),
                                       block_norm='L2-Hys', visualize=True)
    return hog_image

def fun_SIFT(image):
    # SIFT
    sift = cv2.SIFT_create()
    # 使用SIFT检测特征并返回关键点和描述符
    keypoints, descriptors = sift.detectAndCompute(image, None)
    # 绘制关键点
    output_image = cv2.drawKeypoints(image, keypoints, None)
    return output_image


def fun_ORB(image):
    # ORB
    orb = cv2.ORB_create()  # 创建 ORB 特征检测器
    # 检测和提取 ORB 特征
    keypoints1, descriptors = orb.detectAndCompute(image, None)
    output_image1 = cv2.drawKeypoints(image, keypoints1, None)
    # io.imshow(output_image1)
    # io.show()
    return output_image1


plt.subplot(4, 6, 1), plt.imshow(image11), plt.title('Original'), plt.axis('off')
plt.subplot(4, 6, 2), plt.imshow(image12, 'gray'), plt.title('Gray'), plt.axis('off')
plt.subplot(4, 6, 3), plt.imshow(fun_LBP(image12), 'gray'), plt.title('LBP'), plt.axis('off')
plt.subplot(4, 6, 4), plt.imshow(fun_HOG(image12), 'gray'), plt.title('HOG'), plt.axis('off')
plt.subplot(4, 6, 5), plt.imshow(fun_SIFT(image12), 'gray'), plt.title('SIFT'), plt.axis('off')
plt.subplot(4, 6, 6), plt.imshow(fun_ORB(image12), 'gray'), plt.title('ORB'), plt.axis('off')

plt.subplot(4, 6, 7), plt.imshow(image21), plt.title('Original'), plt.axis('off')
plt.subplot(4, 6, 8), plt.imshow(image22, 'gray'), plt.title('Gray'), plt.axis('off')
plt.subplot(4, 6, 9), plt.imshow(fun_LBP(image22), 'gray'), plt.title('LBP'), plt.axis('off')
plt.subplot(4, 6, 10), plt.imshow(fun_HOG(image22), 'gray'), plt.title('HOG'), plt.axis('off')
plt.subplot(4, 6, 11), plt.imshow(fun_SIFT(image22), 'gray'), plt.title('SIFT'), plt.axis('off')
plt.subplot(4, 6, 12), plt.imshow(fun_ORB(image22), 'gray'), plt.title('ORB'), plt.axis('off')

plt.subplot(4, 6, 13), plt.imshow(image31), plt.title('Original'), plt.axis('off')
plt.subplot(4, 6, 14), plt.imshow(image32, 'gray'), plt.title('Gray'), plt.axis('off')
plt.subplot(4, 6, 15), plt.imshow(fun_LBP(image32), 'gray'), plt.title('LBP'), plt.axis('off')
plt.subplot(4, 6, 16), plt.imshow(fun_HOG(image32), 'gray'), plt.title('HOG'), plt.axis('off')
plt.subplot(4, 6, 17), plt.imshow(fun_SIFT(image32), 'gray'), plt.title('SIFT'), plt.axis('off')
plt.subplot(4, 6, 18), plt.imshow(fun_ORB(image32), 'gray'), plt.title('ORB'), plt.axis('off')

plt.subplot(4, 6, 19), plt.imshow(image41), plt.title('Original'), plt.axis('off')
plt.subplot(4, 6, 20), plt.imshow(image42, 'gray'), plt.title('Gray'), plt.axis('off')
plt.subplot(4, 6, 21), plt.imshow(fun_LBP(image42), 'gray'), plt.title('LBP'), plt.axis('off')
plt.subplot(4, 6, 22), plt.imshow(fun_HOG(image42), 'gray'), plt.title('HOG'), plt.axis('off')
plt.subplot(4, 6, 23), plt.imshow(fun_SIFT(image42), 'gray'), plt.title('SIFT'), plt.axis('off')
plt.subplot(4, 6, 24), plt.imshow(fun_ORB(image42), 'gray'), plt.title('ORB'), plt.axis('off')

plt.show()
