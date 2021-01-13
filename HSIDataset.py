'''数据集类'''
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import rotate_matrix_90, flip_from_left2right


class HSIDataset(Dataset):
    def __init__(self, data, label, patchsz=5, is_train=True):
        '''
        :param data: [h, w, bands]
        :param label: [h, w]
        :param patchsz: scale
        '''
        super(HSIDataset, self).__init__()
        # 数据类型转换
        if data.dtype != np.float32: data = data.astype(np.float32)
        if label.dtype != np.int32: label = label.astype(np.int32)
        self.patchsz = patchsz
        # 添加镜像
        data = self.addMirror(data)
        # 数据归一化并缩放到[-1, 1]
        data = 2 * self.Normalize(data) - 1
        # 生成样本和标签
        self.data, self.label = self.generate(data, label)
        if is_train: self.augment()

    def augment(self):
        s, c = self.data.shape[0], self.data.shape[3]
        augment_data = np.zeros((8*s, self.patchsz, self.patchsz, c), dtype=np.float32)
        for i, x in enumerate(self.data):
            for j in range(8):
                if j == 4: x = flip_from_left2right(x)
                augment_data[8*i+j] = x
                x = rotate_matrix_90(x)
        self.data = augment_data
        self.label = np.repeat(self.label, 8)


    def generate(self, data, label):
        # 转换数据格式
        indices = list(zip(*np.nonzero(label)))
        sample = np.zeros((len(indices), self.patchsz, self.patchsz, data.shape[-1]), dtype=np.float32)
        for i, (x, y) in enumerate(indices):
            sample[i] = data[x:x + self.patchsz, y:y + self.patchsz]
        indices = tuple(zip(*indices))
        # 原始标签从1开始计数
        label = label[indices] - 1
        return sample, label

    def __len__(self):
        return self.data.shape[0]

    # 数据归一化
    def Normalize(self, data):
        h, w, c = data.shape
        data = data.reshape((h * w, c))
        data -= np.min(data, axis=0)
        data /= np.max(data, axis=0)
        data = data.reshape((h, w, c))
        return data

    # 添加镜像
    def addMirror(self, data):
        dx = self.patchsz // 2
        h, w, bands = data.shape
        mirror = None
        if dx != 0:
            mirror = np.zeros((h + 2 * dx, w + 2 * dx, bands))
            mirror[dx:-dx, dx:-dx, :] = data
            for i in range(dx):
                # 填充左上部分镜像
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                # 填充右下部分镜像
                mirror[:, -i - 1, :] = mirror[:, -(2 * dx - i) - 1, :]
                mirror[-i - 1, :, :] = mirror[-(2 * dx - i) - 1, :, :]
        return mirror


    def __getitem__(self, index):
        '''
        :param index:
        :return: 光谱信息， 标签
        '''
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.label[index], dtype=torch.long)

class DatasetInfo(object):
    info = {'PaviaU': {
        'data_key': 'paviaU',
        'label_key': 'paviaU_gt'
    },
        'Salinas': {
            'data_key': 'salinas_corrected',
            'label_key': 'salinas_gt'
        },
        'KSC': {
            'data_key': 'KSC',
            'label_key': 'KSC_gt'
    },  'Houston':{
            'data_key': 'Houston',
            'label_key': 'Houston2018_gt'
    },  'Indian':{
            'data_key': 'indian_pines_corrected',
            'label_key': 'indian_pines_gt'
    },  'Pavia':{
            'data_key': 'pavia',
            'label_key': 'pavia_gt'
    }}


# from scipy.io import loadmat
# m = loadmat('data/KSC/KSC.mat')
# data = m['KSC']
# m = loadmat('data/KSC/KSC_gt.mat')
# label = m['KSC_gt']
# dataset = HSIDataset(data, label)
# spectra, gt = dataset[0]
# spectra_3, gt = dataset[3]
# indices = list(zip(*np.nonzero(label)))
# i, j = indices[0]
# if data.dtype != np.float32: data = data.astype(np.float32)
# x = torch.tensor(data[i, j], dtype=torch.float32)
# print(torch.equal(x, spectra[2, 2]))
# print(torch.equal(spectra, torch.tensor(rotate_matrix_90(spectra_3.numpy()))))

