# -*- coding = utf-8 -*-
# @Time :2024/6/3 17:47
# @Author:sls
# @FIle:self-attention.py
# @Annotation: https://mp.weixin.qq.com/s?__biz=Mzg5NDY4ODA0OQ==&mid=2247536682&idx=3&sn=ba818bd5f6f7ce5d73ad9119c72693f7&chksm=c019de74f76e576249404095d074faef14bd31d8241b1e995f3e6d66efa6bbde5efb24babf3f&scene=27

import torch

x = [
    [1, 0, 1, 0],  # Input 1
    [0, 2, 0, 2],  # Input 2
    [1, 1, 1, 1]  # Input 3
]
x = torch.tensor(x, dtype=torch.float32)
