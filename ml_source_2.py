import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torch.autograd import Variable
from torchvision.datasets import CIFAR10


class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()

        # 1. Dummy
        # 28x28 24x24 12x12 8x8 4x4
        # 32x32 28x28 14x14 10x10 5x5
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 30, 5)
        self.lc = nn.Linear(5 * 5 * 30, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 5 * 5 * 30)
        x = self.lc(x)
        # x = sigmoid_(x)

        return x


if __name__ == "__main__":

    print('Starting...')

    lr = 0.01  # learning rate
    momentum = 0.9
    batch_size = 100

    model = net()
    sigmoid_ = nn.Sigmoid()
    criterion = nn.NLLLoss()  # loss
    # CrossEntropyLoss
    # CrossEntropyWithLogits
    # NLLLoss
    # BCELoss
    # BCEWithLogitsLoss
    optimizer = optim.SGD(net.parameters(model), lr, momentum)  # optimizer
    # print(model)

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    # train_set = MNIST(root='./data', train=True, download=True, transform=trans)
    # test_set = MNIST(root='./data', train=False, download=True, transform=trans)

    train_set = CIFAR10(root='./data', train=True, download=True, transform=trans)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=trans)

    # CIFAR10 выборка
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)

    print('==>>> total training batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))

    for epoch in range(10):
        for batch_idx, (x, target) in enumerate(train_loader):  # reading train data
            optimizer.zero_grad()
            x, target = Variable(x), Variable(target)
            # one hot encode
            # target = F.one_hot(target)  # works w/o this

            y = model(x)
            # print('size y : {}  size target : {}'.format(y.size(), target.size()))
            # print(sigmoid_(y).type())
            # print(target)

            loss = criterion(y, target)  # loss
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0 and 0:
                print('==>>> epoch: {}, index: {}'.format(epoch, batch_idx))

        # ==================================================================
        # Testing
        total_cnt = 0
        correct_cnt = 0
        test_loss = 0
        for batch_idx, (x, target) in enumerate(test_loader):  # reading test data
            y = model(x)
            # one_hot_target = F.one_hot(target)
            # print('size y : {}  size target : {}'.format(y.size(), target.size()))
            loss = criterion(y, target)

            test_loss += loss.item()
            _, predict = y.max(1)
            total_cnt += target.size(0)
            correct_cnt += predict.eq(target).sum().item()

            # what
            # _, pred_label = torch.max(y.data, 1)
            # total_cnt += x.data.size()[0]
            # correct_cnt += (pred_label == target.data).sum()

            if batch_idx % 100 == 0:
                print('==>>> epoch: {}, index: {}, acc: {:.2f}, correct: {}, total: {}'.format(
                    epoch, batch_idx, (correct_cnt * 1.) / total_cnt, correct_cnt, total_cnt))

    # torch.save(model.state_dict(), 'model.pt', _use_new_zipfile_serialization=False)

    print("Done!")
