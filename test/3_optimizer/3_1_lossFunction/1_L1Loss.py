# coding: utf-8

import torch
import torch.nn as nn

# ----------------------------------- L1 Loss

# 平均绝对误差
# 生成网络输出 以及 目标输出
output = torch.ones(2, 2, requires_grad=True) * 0.5
target = torch.ones(2, 2)

# 设置三种不同参数的L1Loss
# reduce_False = nn.L1Loss(size_average=True, reduce=False)
# reduce_False = nn.L1Loss(size_average=False, reduce=False)
reduce_False = nn.L1Loss(reduction='none')
# size_average_True = nn.L1Loss(size_average=True, reduce=True)
size_average_True = nn.L1Loss(reduction='mean')
# size_average_False = nn.L1Loss(size_average=False, reduce=True)
size_average_False = nn.L1Loss(reduction='sum')

print('target', target)
o_0 = reduce_False(output, target)
print('output1', output)
o_1 = size_average_True(output, target)
print('output2', output)
o_2 = size_average_False(output, target)
print('output3', output)

print('\nreduce=False, 输出同维度的loss:\n{}\n'.format(o_0))
print('size_average=True，\t求平均:\t{}'.format(o_1))
print('size_average=False，\t求和:\t{}'.format(o_2))

# loss = nn.L1Loss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5)
# output = loss(input, target)
# output.backward()
#
# print('input', input, input.grad)
# print('output', output)
