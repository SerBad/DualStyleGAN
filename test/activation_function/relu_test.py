import torch
import torch.nn as nn
import torch.nn.functional as F

# https://cloud.tencent.com/developer/article/1797190
input = torch.tensor([-1, -10, 100, 1])
# max(0,x)
output1 = F.relu(input)
# inplace表示是否直接修改输入的值
# output2 = F.relu(input, inplace=True)
print(output1)
# print(input)
