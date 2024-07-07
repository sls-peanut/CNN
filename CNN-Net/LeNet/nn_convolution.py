# -*- coding = utf-8 -*-
# @Time :2023/5/1 20:10
# @Author:sls
# @File :nn_convolution.py
# @Software:PyCharm
import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# print(input.shape)
# print(kernel.shape)

input = torch.reshape(input, (1, 1, 5, 5,))
kernel = torch.reshape(kernel, (1, 1, 3, 3,))

print(input.shape)
print(kernel.shape)

# 输出=输入-kernel_size/stride +1
output = F.conv2d(input, kernel, stride=1)
print(output)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
