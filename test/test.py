import torch
import math
import torch.nn as nn
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union


# 拼接的方法
def cat():
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


# https://zhuanlan.zhihu.com/p/83172023
def test_tensor():
    # 导数计算公式
    # https://zs.symbolab.com/solver/derivative-specific-methods-calculator

    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    # ** 表示乘方，也就是次方
    z = 2 * x ** 3 + y
    # torch.autograd.backward(z) 和 z.backward() 等价
    z.backward()
    # 得到的x.grad是6，是因为3次方的导数是3，乘以常数2，就是6
    # y.grad是1
    print(z, x, x.grad_fn, x.grad, y.grad)

    print(torch.ones(2, 3))
    print(torch.rand(2, 3, 4, 5))


def grad_tensor():
    # https://blog.csdn.net/u010414589/article/details/115216784
    # 标量就是一个数字。标量也称为0维数组。
    # 向量是一组标量组成的列表。向量也称为1维数组。

    # x = torch.ones(2, 2, requires_grad=True)
    # x = torch.rand(2, 2, 5, requires_grad=True)
    x = torch.tensor(([[2., 2.], [4., 3.]]), requires_grad=True)
    z = x ** 3 + 2
    # view(1, 4)转成1*4的矩阵，需要和原本的数量保持一致
    # view中值为-1表示直接转成一维张量
    print(z.view(1, 4))
    # z.sum().backward()
    # 张量不能使用backward()，需要转化为标量，
    z.backward(torch.ones_like(x))
    print(z, x, x.grad, sep='\n')


def mean_test():
    # e表示表示10的次方
    print(1e-8 - 10 ** -8, '2e2', 2e2)
    # 1除以torch.tensor(4)的开根号
    print(torch.rsqrt(torch.tensor(4)))

    x = torch.Tensor([1, 2, 3, 4, 5, 6]).view(2, 3)
    # 对dim维求平均，
    y_0 = torch.mean(x, dim=0)
    y_1 = torch.mean(x, dim=1)
    # 如果要保持维度不变(例如在深度网络中)，则可以加上参数keepdim=True：
    y_0_0 = torch.mean(x, dim=1, keepdim=True)
    y_0_1 = torch.mean(x, dim=1, keepdim=False)
    print(x)
    print(y_0)
    print(y_1)
    print(y_0_0)
    print(y_0_1)
    # 操作符 // ，以执行地板除：//除法不管操作数为何种数值类型，总是会舍去小数部分，返回数字序列中比真正的商小的最接近的数字。
    print("6 // 2 * 2", 6.0 // 5.0)
    print("math.log(size, 2)", math.log(1024, 2), " 2 ** 10 ", 2 ** 10)
    print("torch.eye(512)", torch.eye(512) * 512.0 ** 0.5)

    print("torch.tensor(True)", torch.tensor(True).item())


def for_tes():
    data = [3, 7, 9, 5, 6, 7, 6, 1]

    # Union[int, str] 表示既可以是int，也可以str
    # https://docs.python.org/zh-cn/3/library/typing.html#typing.Union
    def add(s: int) -> Union[int, str, nn.Sequential]:
        return s + s

    # for循环的一种简单方式
    result = [add(s) for s in data]

    result.append(11)
    for s in result:
        print("s1", s)

    result[-1] = 10
    for s in result:
        print("s2", s)

    print(math.log(512, 2), "res_index // 2 * 2", 6 // 2 * 2)


def test_reshape():
    a = torch.rand(1, 3, 1024, 1024)
    a_shape = a.shape
    b = a.reshape(a_shape[0] * a_shape[1], a_shape[2])
    print(a_shape, b.shape)


def test_tensor_operation():
    a = torch.tensor([1.0, 2.0, 3.0])
    print("原始的", a)
    # 除法
    a.div_(0.5)
    print("div_后", a)
    a = torch.div(a, 0.5)
    print("torch.div_后", a)
    a.fill_(2.0)
    print("fill__后", a)
    a = torch.fill_(a, 3.0)
    print("torch.fill后", a)

if __name__ == "__main__":
    # cat()
    # test_tensor()
    # grad_tensor()
    # mean_test()
    # for_tes()
    # test_reshape()
    test_tensor_operation()
