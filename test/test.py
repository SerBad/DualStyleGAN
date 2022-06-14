import torch

x = torch.rand(2, 3)
y = torch.rand(4, 3)
z = torch.rand(2, 9)

print(x, y, z)

print(torch.cat((x, y), dim=0))
print(torch.cat((x, z), dim=1))
