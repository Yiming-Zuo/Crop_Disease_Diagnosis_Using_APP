import argparse
from model.backbone.MobileViT import *
import mixnet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import datasets, transforms, models

import os
import argparse
import logging
import numpy as np
import micro
import random

from utils import progress_bar
from visualdl import LogWriter

import light_cnns


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(666)

train_loader = None
test_loader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0
start_epoch = 1

def prepare(args):
    global train_loader
    global test_loader
    global net
    global criterion
    global optimizer

    print('==> Preparing data..')
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_size = 0.3
    # 读取数据集
    data = datasets.ImageFolder(root='./datasets', transform=data_transform)
    # 打乱数据集索引，分为训练集索引和验证集集索引
    num_data = len(data)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    split = int(np.floor(num_data * val_size))
    train_idx, val_idx = indices[split:], indices[:split]  # 分割索引号字典
    # 采样器
    train_sampler = SubsetRandomSampler(train_idx)  # 根据下标随机采样
    val_sampler = SubsetRandomSampler(val_idx)
    # 加载测试集
    train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    test_loader = torch.utils.data.DataLoader(data, batch_size=args.test_batch_size, sampler=val_sampler, num_workers=4)


    print('==> Building model..')
    num_classes =61
    if args.model == 'vgg':
        net = models.vgg19(pretrained=True)
        net.classifier._modules['6'] = nn.Linear(4096, 61)
    # if args.model == 'resnet18':
    #     net = ResNet18()
    # if args.model == 'googlenet':
    #     net = GoogLeNet()
    # if args.model == 'densenet121':
    #     net = DenseNet121()
    # if args.model == 'mobilenet':
    #     net = MobileNet()
    if args.model == 'mobilenetv2':
        net = models.mobilenet_v2(pretrained=True)
        net.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 61),
        )
    # if args.model == 'shufflenetg2':
    #     net = ShuffleNetG2()
    # if args.model == 'senet18':
    #     net = SENet18()
    if args.model == 'mobilevit_xxs':
        net = mobilevit_xxs()
    if args.model == "micronet":
        net = micro.micronet(input_size=256, num_classes=61)
    if args.model == "mixnet":
        net = mixnet.MixNet(input_size=256, num_classes=61)
    if args.model == "parnet":
        net = light_cnns.parnet_s(3, n_classes=61)

    print(net)

    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    if args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=True,weight_decay=args.weight_decay)
    if args.optimizer == 'Adamax':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    return net, scheduler


def train(epoch):
    global train_loader
    global test_loader
    global net
    global criterion
    global optimizer

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = correct/total

        progress_bar(batch_idx, len(train_loader), 'epoch: %d | lr: %f | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (epoch, scheduler.get_last_lr()[-1], train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), correct/total


def test(epoch, args):
    global best_acc
    
    global train_loader
    global test_loader
    global net
    global criterion
    global optimizer

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = correct/total
            loss__ = test_loss / (batch_idx + 1)

            progress_bar(batch_idx, len(test_loader), 'epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        filepath = os.path.join('checkpoint', "{}-{}.pt".format(args.model, args.optimizer))
        torch.save(state, filepath)
        best_acc = acc
    return test_loss/(batch_idx+1), acc, best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model", type=str, default="mobilenetv2")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=-1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--load", type=bool, default=False)

    args = parser.parse_args()

    try:
        writer = LogWriter(logdir="./log/scalar")

        _logger = logging.getLogger()
        _logger.setLevel(logging.INFO)
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        log_path = os.path.join('logs', "{}.log".format(args.model))
        fh = logging.FileHandler(log_path, mode='a')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        _logger.addHandler(fh)

        _logger.info("model:{}, lr:{}, optimizer:{}, epochs:{}".format(args.model, args.lr, args.optimizer, args.epochs))

        net, scheduler = prepare(args)
        acc = 0.0
        best_acc = 0.0
        if args.load:
            print("==> load checkpoint..")
            filepath_load = os.path.join('checkpoint', "{}-{}.pt".format(args.model, args.optimizer))
            stata_load = torch.load(filepath_load)
            net.load_state_dict(stata_load['net'])
            start_epoch = stata_load['epoch'] + 1
            best_acc = stata_load['acc']
            print("------load start_epoch:{}, best_acc:{}------".format(start_epoch, best_acc))
            _logger.info("------load start_epoch:{}, best_acc:{}------".format(start_epoch, best_acc))
        for epoch in range(start_epoch, start_epoch+args.epochs):
            loss_train, acc_train = train(epoch)
            writer.add_scalar(tag=args.model + "train/loss", step=epoch, value=loss_train)
            writer.add_scalar(tag=args.model + "train/acc", step=epoch, value=acc_train)

            loss_test, acc, best_acc = test(epoch, args)
            writer.add_scalar(tag=args.model + "test/loss", step=epoch, value=loss_test)
            writer.add_scalar(tag=args.model + "test/acc", step=epoch, value=acc)

            _logger.info("epoch:{}，lr:{}, acc:{}, best_acc: {}".format(epoch, scheduler.get_last_lr()[-1], acc, best_acc))
            if args.gamma > 0:
                scheduler.step()

        _logger.info("best_acc:" + str(best_acc))
        writer.close()
    except Exception as exception:
        _logger.exception(exception)
        raise