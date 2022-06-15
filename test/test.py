import torch


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


def grad_tensor():
    # x = torch.ones(2, 2, requires_grad=True)
    # x = torch.rand(2, 2, 5, requires_grad=True)
    x = torch.tensor(([[2., 2.], [4., 3.]]), requires_grad=True)
    z = x ** 3 + 2
    # 转成一维数据
    print(z.view(1, 4))
    # z.sum().backward()
    # 张量不能使用backward()，需要转化为标量，
    z.backward(torch.ones_like(x))
    print(z, x, x.grad, sep='\n')


if __name__ == "__main__":
    # cat()
    # test_tensor()
    grad_tensor()
