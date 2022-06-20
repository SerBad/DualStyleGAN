# coding: utf-8

import torch
import torch.nn as nn

# ----------------------------------- MSE loss
# 均方损失函数 (x-y)²
# 生成网络输出 以及 目标输出
output = torch.ones(2, 2, requires_grad=True) * 0.5
target = torch.ones(2, 2)

# 设置三种不同参数的MSELoss
# reduce_False = nn.MSELoss(size_average=True, reduce=False)
# reduce_False = nn.MSELoss(size_average=False, reduce=False)
reduce_False = nn.MSELoss(reduction='none')
# size_average_True = nn.MSELoss(size_average=True, reduce=True)
size_average_True = nn.MSELoss(reduction='mean')
# size_average_False = nn.MSELoss(size_average=False, reduce=True)
size_average_False = nn.MSELoss(reduction='sum')

o_0 = reduce_False(output, target)
o_1 = size_average_True(output, target)
o_2 = size_average_False(output, target)

print('output', output)
print('target', target)
print('\nreduce=False, 输出同维度的loss:\n{}\n'.format(o_0))
print('size_average=True，\t求平均:\t{}'.format(o_1))
print('size_average=False，\t求和:\t{}'.format(o_2))
