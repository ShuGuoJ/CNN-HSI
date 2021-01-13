import torch
from torch.nn import init
from torch import nn
import os
from scipy.io import loadmat
import random
import numpy as np


def weight_init(m):
    if isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 5e-2)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        init.normal_(m.weight, 0, 5e-2)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def loadLabel(path):
    '''
    :param path:
    :return: 训练样本标签， 测试样本标签
    '''
    assert os.path.exists(path), '{},路径不存在'.format(path)
    # keys:{train_gt, test_gt}
    gt = loadmat(path)
    return gt['train_gt'], gt['test_gt']


def splitSampleByClass(gt, ratio, seed=971104):
    '''
    :param gt: 样本标签
    :param ratio: 随机抽样每类样本的比例
    :param seed: 随机种子
    :return: 训练样本， 测试样本
    '''
    # 设置随机种子
    random.seed(seed)
    train_gt = np.zeros_like(gt)
    test_gt = np.copy(gt)
    train_indices = []
    nc = int(np.max(gt))
    # 开始随机挑选样本
    for c in range(1, nc + 1):
        samples = np.nonzero(gt == c)
        sample_indices = list(zip(*samples))
        size = int(len(sample_indices) * ratio)
        x = random.sample(sample_indices, size)
        train_indices += x
    indices = tuple(zip(*train_indices))
    train_gt[indices] = gt[indices]
    test_gt[indices] = 0
    return train_gt, test_gt


# 模型训练
def train(model, criterion, optimizer, dataLoader, device):
    '''
    :param model: 模型
    :param criterion: 目标函数
    :param optimizer: 优化器
    :param dataLoader: 批数据集
    :return: 已训练的模型，训练损失的均值
    '''
    model.train()
    model.to(device)
    trainLoss = []
    for step, (input, target) in enumerate(dataLoader):
        input, target = input.to(device), target.to(device)
        input = input.permute((0, 3, 1, 2))
        out = model(input)
        out = out.squeeze(-1).squeeze(-1)
        loss = criterion(out, target)
        trainLoss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%5 == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('step:{} loss:{} lr:{}'.format(step, loss.item(), lr))
    return model, float(np.mean(trainLoss))


# 模型测试
def test(model, criterion, dataLoader, device):
    model.eval()
    model.to(device)
    evalLoss, correct = [], 0
    for input, target in dataLoader:
        input, target = input.to(device), target.to(device)
        input = input.permute((0, 3, 1, 2))
        logits = model(input)
        logits = logits.squeeze(-1).squeeze(-1)
        loss = criterion(logits, target)
        evalLoss.append(loss.item())
        pred = torch.argmax(logits, dim=-1)
        correct += torch.sum(torch.eq(pred, target).int()).item()
    acc = float(correct) / len(dataLoader.dataset)
    return acc, np.mean(evalLoss)


'''矩阵逆时针旋转90度'''
def rotate_matrix_90(m):
    assert len(m.shape) >= 2
    h, w = m.shape[:2]
    ans = np.zeros_like(m)
    new_shape = list(range(len(m.shape)))
    new_shape[0], new_shape[1] = new_shape[1], new_shape[0]
    ans = ans.transpose(new_shape)
    x_coor = np.arange(h).reshape((h, 1))
    y_coor= -np.arange(w).reshape((1, w)) + w - 1
    x_coor, y_coor = np.repeat(x_coor, w, 1), np.repeat(y_coor, h, 0)
    ans[y_coor, x_coor] = m
    return ans


'''矩阵水平翻转'''
def flip_from_left2right(m):
    assert len(m.shape) >= 2
    h, w = m.shape[:2]
    ans = np.zeros_like(m)
    x_coor = np.arange(h).reshape((h, 1))
    y_coor = w - np.arange(w).reshape((1, w)) - 1
    x_coor, y_coor = np.repeat(x_coor, w, 1), np.repeat(y_coor, h, 0)
    ans[x_coor, y_coor] = m
    return ans
