import torch
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)  # 构造一段连续的数据
x = Variable(x)  # 转换成张量
x_np = x.data.numpy()  # plt中形式需要numoy形式，tensor形式会报错


def relu():
    # https://cloud.tencent.com/developer/article/1797190
    input = torch.tensor([-1, -10, 100, 1])
    # max(0,x)
    output1 = F.relu(input)
    # inplace表示是否直接修改输入的值
    # output2 = F.relu(input, inplace=True)
    print(output1)
    # print(input)


def softplus():
    # https://www.cnblogs.com/carrollCN/p/11370960.html
    y_softplus = F.softplus(x).data.numpy()
    plt.plot(x_np, y_softplus, c='red', label='softplus')
    plt.ylim((-0.2, 6))
    plt.legend(loc='best')

    plt.show()


if __name__ == "__main__":
    softplus()
