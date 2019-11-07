
import torch

from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms, datasets
from torchvision import models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler, optimizer
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset,DataLoader
import time


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



def fit(epoch, model, data_loader, phase='training', volatile=False):
    optimizer = optim.SGD(model.parameters(),lr = 0.01,momentum=0.5)

    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):

        data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        running_loss += F.nll_loss(output, target, size_average=False).data
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy
def print_and_plot(model,train_loader,test_loader):
    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    for epoch in range(1, 20):
        epoch_loss, epoch_accuracy = \
            fit(epoch, model, train_loader, phase='training', volatile=False)
        val_epoch_loss, val_epoch_accuracy = \
            fit(epoch, model, test_loader, phase='validation', volatile=False)
        train_losses.append(epoch_loss)
        print(f'train_loss:\n{train_losses}')
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
    plt.figure(1)
    # plot the trainging and test loss
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo', label='training loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r', label='validation loss')
    plt.legend()
    plt.savefig('MNIST' + str(epoch) + '.jpg')
    plt.figure(2)
    # plots the training and test accuracy
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'bo', label='training accuracy')
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, 'r', label='val accuracy')
    plt.legend()
    plt.savefig('MNIST_accuracy')

    pass

def main():
    dir = '/home/ZhangXueLiang/LiMiao/dataset/Mnist'
    model = Net()
    transformation = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (
                                             0.3081,))])  # https://stackoverflow.com/questions/56745486/pytorch-dataloader-indexerror-too-many-indices-for-tensor-of-dimension-0/56748549
    train_dataset = datasets.MNIST(dir, train=True, transform=transformation, download=True)
    test_dataset = datasets.MNIST(dir, train=False, transform=transformation, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    print_and_plot(model, train_loader, test_loader)
    # view_()
    # variable_()
    # type_()
    # enumerate_()
    pass
def enumerate_():
    seasons = ['Spring','Summer','Fall','Winter']
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    list_seasons = list(enumerate(seasons))
    list_months = list(enumerate(months))
    print(f'list_seasons:\n{list_seasons}')
    print(f'list_months:\n{list_months}')
    for i,month in enumerate(months):
        print(f'i:{i+1} month:{month}')

    pass

def variable_():

    # # 对数求导
    # x = Variable(torch.Tensor([1]), requires_grad = True)
    # w = Variable(torch.Tensor([3]), requires_grad = True)
    # b = Variable(torch.Tensor([3]), requires_grad = True)
    # # Bulid a computational graph
    # y = w * x + b # y = 2*x +3
    # y.backward()  #y.backward()这一行代码就是所谓的自动求导,直接通过这个就可以对所有需要梯度的变量进行求导,得到他们的梯度
    # print(f'x.grad:\n{x.grad}')
    # print(f'w.grad:\n{w.grad}')
    # print(f'b.grad:\n{b.grad}')

    # 对矩阵求导
    x = torch.randn(3)
    x = Variable(x,requires_grad = True)
    w = Variable(torch.randn(3),requires_grad  =True)
    y = w.mul(x)  # torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
                  # torch.mm(a, b)是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
    print(f'x:\n{x}')
    print(f'w:\n{w}')
    print(f'y:\n{y}')
    y.backward(torch.Tensor([1,0.1,0.01]))   # y.backward(torch.Tensor([1,0.1,0.01])) ,得到的梯度是他们原有的梯度乘以1 ,0.1 和0.01.
    print(f'x.grad:\n{x.grad}')
    print(f'w.grad:\n{w.grad}')


    pass

def type_():
    print(f'type((0.5))\n{type((0.5))}')
    print(f'type((0.5,)):\n{type((0.5,))}')
    pass

def view_():
    x = torch.randn(2 ,2 ,2)
    print(f'x:\n{x}')
    y = x.view(-1 ,1, 8)
    print(f'y:\n{y}')
    pass
if __name__ == '__main__':
    main()
