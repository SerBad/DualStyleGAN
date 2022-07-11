import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch

import argparse
from argparse import Namespace
from torchvision import transforms

import torchvision
import onnx
# import onnxruntime
import netron


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # 全连接层，三个参数分别是，输入通道，输出通道，卷积核大小
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        # 全连接层
        # https://blog.csdn.net/qq_42079689/article/details/102873766
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


torch.set_printoptions(profile='full')
net = Net()
input = torch.randn(1, 1, 32, 32)

print(net)
print("input", input.size())
print("net(input)", net(input))
path = "./main.onnx"
torch.onnx.export(net,
                  input,
                  path,
                  verbose=True,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=["input"],
                  output_names=["output"],
                  keep_initializers_as_inputs=True)

print(onnx.checker.check_model(onnx.load(path)))
netron.start(path)
# onnx.save(onnx.shape_inference.infer_shapes(path), onnx.load(path))

# session = onnxruntime.InferenceSession(path)
# print("session.get_inputs()", session.get_inputs())
# for o in session.get_inputs():
#     print(o)
# for o in session.get_outputs():
#     print("session.get_outputs()", o)
