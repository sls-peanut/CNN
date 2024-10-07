# -*- coding = utf-8 -*-
# @Time :2024/10/2 16:05
# @Author:sls
# @FIle:shaoShuai.py
# @Annotation:
import os
import time


def clear_screen():
    """清屏函数，根据操作系统选择清屏命令"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_plane(position):
    """打印飞机在指定位置"""
    print(" " * position + "✈️")


def print_guards_and_shao_shuai(offset):
    """打印卫兵和少帅并添加偏移行数"""
    print("\n" * offset + "  👮    少帅    👮")  # 使用通用字符表示卫兵


def main():
    plane_position = 0  # 飞机的起始位置
    width = 30  # 屏幕宽度

    # 清屏
    clear_screen()

    # 飞机移动到中间位置
    for i in range(width // 2):
        clear_screen()
        print_plane(plane_position)  # 打印飞机
        time.sleep(0.1)  # 控制移动速度
        plane_position += 1  # 向右移动

    # 飞机到达中间位置后，打印飞机
    clear_screen()
    print_plane(plane_position)  # 打印飞机
    time.sleep(2)  # 等待一段时间，模拟飞机降落

    # 逐步换行，打印卫兵和少帅
    for i in range(5):
        clear_screen()
        print_plane(plane_position)  # 打印飞机
        print_guards_and_shao_shuai(i)  # 打印卫兵和少帅
        time.sleep(1)  # 每次换行延迟1秒


if __name__ == "__main__":
    main()