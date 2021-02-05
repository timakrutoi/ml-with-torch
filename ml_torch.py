import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable
from torchvision.datasets import CIFAR10

import numpy as np


class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()

        # 1. Dummy
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

        return x


def BCELossWrapper(f=nn.BCELoss()):
    def wrapper(output, target):
        return f(sigmoid_(output), F.one_hot(target).float())

    return wrapper


def BCEWithLogitsLossWrapper(f=nn.BCEWithLogitsLoss()):
    def wrapper(output, target):
        return f(output, F.one_hot(target).float())

    return wrapper


def NLLLossWrapper(f=nn.NLLLoss()):
    def wrapper(output, target):
        return f(nn.functional.log_softmax(output, dim=1), target)

    return wrapper


losses = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'BCELoss': BCELossWrapper,
    'BCEWithLogitsLoss': BCEWithLogitsLossWrapper,
    'NLLLoss': NLLLossWrapper
}


class LossFactory():
    def __call__(self, loss_type='CrossEntropyLoss'):
        return losses[loss_type]()


if __name__ == "__main__":

    # tensorboard --logdir=runs

    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    parser = argparse.ArgumentParser(description='loss function selection')
    parser.add_argument('-a', action='store', dest='loss_type', default='CrossEntropyLoss')
    args = parser.parse_args()

    print('Starting...')
    print('Selected loss function: {}'.format(args.loss_type))

    lr = 0.01  # learning rate
    momentum = 0.9
    batch_size = 100

    model = net()
    sigmoid_ = nn.Sigmoid()
    criterion = LossFactory()(args.loss_type)

    writer = SummaryWriter()  # tensorboard

    optimizer = optim.SGD(net.parameters(model), lr, momentum)  # optimizer

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

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
        train_iterator = tqdm(train_loader, ncols=100, desc='Epoch: {}, training'.format(epoch))
        for batch_idx, (x, target) in enumerate(train_iterator):  # reading train data
            optimizer.zero_grad()
            x, target = Variable(x), Variable(target)

            y = model(x)
            loss = criterion(y, target)

            loss.backward()
            optimizer.step()

        train_iterator.close()
        # ==================================================================
        # Testing
        total_cnt = 0
        correct_cnt = 0
        test_loss = 0
        batch_idx = 0
        acc = 0
        test_iterator = tqdm(test_loader, ncols=128, desc='Epoch: {}, testing '.format(epoch))

        for batch_idx, (x, target) in enumerate(test_iterator):  # reading test data
            y = model(x)
            loss = criterion(y, target)

            test_loss += loss.item()
            _, predict = y.max(1)
            total_cnt += target.size(0)
            correct_cnt += predict.eq(target).sum().item()
            acc = (correct_cnt * 1.) / total_cnt
            test_iterator.set_postfix(str='acc: {:.3f}, loss: {:.3f}'.format(epoch, acc, test_loss/(batch_idx + 1)))
            test_iterator.update()

            writer.add_scalar('Acc(test)', acc, batch_idx + (epoch * 100))
            writer.add_scalar('Loss(test)', test_loss, batch_idx + (epoch * 100))

            if batch_idx < 10:
                #  10 * (epoch - 1) < batch_idx < 10 * epoch:
                writer.add_image('Epoch {} :Testing image - label {} : {}'.format(
                    epoch, classes[predict[batch_idx]], classes[target[batch_idx]]), x[batch_idx] + 0.5, 0)

        test_iterator.close()

    torch.save(model.state_dict(), 'calculated models\\model(try2).pt')
    writer.close()

    print("Done!")
