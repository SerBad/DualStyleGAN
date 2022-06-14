import torch

# # cat
# x = torch.rand(2, 3)
# y = torch.rand(4, 3)
# z = torch.rand(2, 9)
#
# print(x, y, z)
#
# # cat 表示拼接函数，dim=0表示是按列来拼接，两个张量的列数必须是一致的
# # dim=1表示是按行来拼接的，两个张量的行数必须是一致的
# print(torch.cat((x, y), dim=0))
# print(torch.cat((x, z), dim=1))

# # stack
a = torch.rand(2, 3, 4)
b = torch.rand(2, 3, 4)
c = torch.rand(2, 3, 4)

print(a)
# stack也是拼接函数，只是是按照维度来拼接的，拼接之后会多一维度，要求张量的行数和列数是一致的
print(torch.stack((a, b, c), dim=0))
print(torch.stack((a, b, c), dim=2))
