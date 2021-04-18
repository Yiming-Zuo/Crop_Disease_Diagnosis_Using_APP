from time import time

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

np.random.seed(0)
start = time()

device = torch.device('cuda')
num_workers = 0
batch_size = 32
val_size = 0.3
n_epochs = 10


# 图像预处理
data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])

# 读取数据集
data = torchvision.datasets.ImageFolder(root='./datasets', transform=data_transform)

torch.cuda.empty_cache()  # 释放显存

# 打乱数据集索引，分为训练集索引和验证集集索引
num_data = len(data)
indices = list(range(num_data))
np.random.shuffle(indices)
split = int(np.floor(num_data * val_size))
train_idx, val_idx = indices[split:], indices[:split]  #分割索引号字典

# 采样器
train_sampler = SubsetRandomSampler(train_idx)       #根据下标随机采样
val_sampler = SubsetRandomSampler(val_idx)

# 根据batch_size划分数据集
train_iter = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
val_iter = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)

# 创建网络
net = torchvision.models.resnet50(pretrained=True)

# 更改网络分类器输出类别
net.fc.out_features = 2

net.cuda()
print(net)

loss = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, )  # 1e-4

# 训练
# 初始化验证集最小误差为正无穷
test_loss_min = np.Inf
# 将训练过程中的训练损失和验证损失存储在列表中
train_loss_list = []
test_loss_list = []

# 导入模型参数
# net.load_state_dict(torch.load('params.pt'))
print('********** 我要开始训练了！ **********')

for epoch in range(n_epochs):
    # 初始化训练损失和验证损失
    train_loss, test_loss = 0.0, 0.0

    net.train()
    # 读取图片序列号和对应的标签值
    for step, (X, y) in enumerate(train_iter):
        X, y = X.to(device), y.to(device)
        output = net(X)
        l = loss(output, y)
        # 清空过往的梯度
        optimizer.zero_grad()
        l.backward()
        # 权重更新
        optimizer.step()
        #求出loss总值
        train_loss += float(l)
        print(X.size(0))

    net.eval()
    with torch.no_grad():  # 防止GPU空间不够
       for step, (X, y) in enumerate(val_iter):
            X, y = X.to(device), y.to(device)
            output = net(X)
            l = loss(output, y)
            test_loss += float(l) * X.size(0)
            print(X.size(0))
            print("-------------")

    # train_acc = utl.evaluate_accuracy(train_iter, net)
    test_acc = utl.evaluate_accuracy(val_iter, net)
    #求平均值
    train_loss = train_loss / len(train_iter.dataset)
    test_loss = test_loss / len(val_iter.dataset)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

    end = time() - start
    print("epoch:{} | train_loss:{:.4f} | val_loss:{:.4f} | val_acc:{:.2%} | time:{:02}:{:02}:{:02}".format(epoch+1,
                                                                                                  train_loss,
                                                                                                  test_loss,
                                                                                                  test_acc,
                                                                                                  int(end // 3600),
                                                                                                  int(end % 3600 // 60),
                                                                                                  int(end % 3600 % 60)))
    # 如果损失降低，保存模型参数
    if test_loss <= test_loss_min:
        print("test_loss: {} >>>>>> {}".format(test_loss_min, test_loss))
        torch.save(net.state_dict(), "params.pt")
        test_loss_min = test_loss
