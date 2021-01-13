'''模型最终训练'''
import torch
from torch import nn, optim
import numpy as np
from scipy.io import loadmat, savemat
from utils import weight_init, loadLabel, train, test
from HSIDataset import HSIDataset, DatasetInfo
from Model.module import CNN_HSI
from torch.utils.data import DataLoader
import os
import argparse
from visdom import Visdom


isExists = lambda path: os.path.exists(path)
SAMPLE_PER_CLASS = [10, 50, 100]
RUN = 10
EPOCHS = 10
LR = 1e-1
BATCHSZ = 16
NUM_WORKERS = 8
SEED = 971104
torch.manual_seed(SEED)
DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
viz = Visdom(port=17000)
ROOT = None
MODEL_NAME = 'CNN_HSI'


def main(datasetName, n_sample_per_class, run):
    # 加载数据和标签
    info = DatasetInfo.info[datasetName]
    data_path = "./data/{}/{}.mat".format(datasetName, datasetName)
    label_path = './trainTestSplit/{}/sample{}_run{}.mat'.format(datasetName, n_sample_per_class, run)
    isExists(data_path)
    data = loadmat(data_path)[info['data_key']]

    bands = data.shape[2]
    isExists(label_path)
    trainLabel, testLabel = loadLabel(label_path)
    res = torch.zeros((3, EPOCHS))
    # 数据转换
    data, trainLabel, testLabel = data.astype(np.float32), trainLabel.astype(np.int), testLabel.astype(np.int)
    nc = int(np.max(trainLabel))
    trainDataset = HSIDataset(data, trainLabel)
    testDataset = HSIDataset(data, testLabel, is_train=False)
    trainLoader = DataLoader(trainDataset, batch_size=BATCHSZ, shuffle=True, num_workers=NUM_WORKERS)
    testLoader = DataLoader(testDataset, batch_size=128, shuffle=True, num_workers=NUM_WORKERS)
    model = CNN_HSI(bands, nc)
    # 初始化模型参数
    model.apply(weight_init)
    # 损失函数，优化器，学习率下架管理器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    for epoch in range(EPOCHS):
        print('*'*5 + 'Epoch:{}'.format(epoch) + '*'*5)
        model, trainLoss = train(model, criterion=criterion, optimizer=optimizer, dataLoader=trainLoader, device=DEVICE)
        acc, evalLoss = test(model, criterion=criterion, dataLoader=testLoader, device=DEVICE)
        print('epoch:{} trainLoss:{:.8f} evalLoss:{:.8f} acc:{:.4f}'.format(epoch, trainLoss, evalLoss, acc))
        print('*'*18)
        res[0][epoch], res[1][epoch], res[2][epoch] = trainLoss, evalLoss, acc
        if epoch%5==0:
            torch.save(model.state_dict(), os.path.join(ROOT, '{}_sample{}_run{}_epoch{}.pkl'.format(MODEL_NAME, n_sample_per_class,
                                                                                                       run, epoch)))
        # scheduler.step()
    tmp = res.numpy()
    savemat(os.path.join(ROOT, 'res.mat'), {'trainLoss':tmp[0], 'evalLoss':tmp[1], 'acc':tmp[2]})
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train twoCnn')
    parser.add_argument('--name', type=str, default='KSC',
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=1,
                        help='模型的训练次数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')

    args = parser.parse_args()
    EPOCHS = args.epoch
    datasetName = args.name
    LR = args.lr
    print('*'*5 + datasetName + '*'*5)
    for i, num in enumerate(SAMPLE_PER_CLASS):
        print('*' * 5 + 'SAMPLE_PER_CLASS:{}'.format(num) + '*' * 5)
        res = torch.zeros((RUN, 3, EPOCHS))
        for r in range(RUN):
            print('*' * 5 + 'run:{}'.format(r) + '*' * 5)
            ROOT = '{}/{}/{}/{}'.format(MODEL_NAME, datasetName, num, r)
            if not os.path.isdir(ROOT):
                os.makedirs(ROOT)
            res[r] = main(datasetName, num, r)
        mean = torch.mean(res, dim=0) #[3, EPOCHS]
        viz.line(mean.T, list(range(EPOCHS)), win='{}_SAMPLE_PER_CLASS_{}'.format(MODEL_NAME, num), opts=dict(title='{}_SAMPLE_PER_CLASS_{}'.format(MODEL_NAME, num),
                                                                                               legend=['train loss', 'test loss', 'acc']))
    print('*'*5 + 'FINISH' + '*'*5)