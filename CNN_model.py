import torch
from distributed import Variable
# from pylint.test.functional.useless_object_inheritance import F
import torch.nn.functional as F

from torch import nn
import uuid

# from torch.autograd import variable
from torch.autograd import variable


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        pass

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def one_stride():
    conv = nn.Conv1d(2, 2, 3, bias=False)          # 输出通道增加后，就弄不懂了，呃呃呃呃呃
    sample = torch.randn(2, 2, 7)
    print(f'sample:\n{sample}')
    print(conv.weight)
    # v_sample = conv(Variable(sample))       # 这里不能加Variable，否则出错：RuntimeError: bool value of Tensor with more than one value is ambiguous
    v_sample = conv(sample)       # 这里不能加Variable，否则出错：RuntimeError: bool value of Tensor with more than one value is ambiguous
    print(f'v_sample:\n{v_sample}')
    # v2_sample = conv(variable(sample))    # 但是可以加variable，但会有警告，显示：warnings.warn("torch.autograd.variable(...) is deprecated, use torch.tensor(...) instead")
    # print(f'v2_samole\n{v2_sample}')
    # print(f'uuid.uuid4:{uuid.uuid4()}')     # uuid 通用唯一标识符
    pass

def main():
    one_stride()


    pass
if __name__ == '__main__':
    main()