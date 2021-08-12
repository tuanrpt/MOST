#  Copyright (c) 2021, Tuan Nguyen.
#  All rights reserved.

import os

import numpy as np
from scipy.io import loadmat
from keras.utils.np_utils import to_categorical
from generic_utils import random_seed


def load_mat_file_single_label(filename):
    filename_list = ['mnist', 'stl32', 'synsign', 'gtsrb', 'cifar32', 'usps32']
    data = loadmat(filename)
    x = data['X']
    y = data['y']
    if any(fn in filename for fn in filename_list):
        if 'mnist32_60_10' not in filename:
            y = y[0]
        else:
            y = np.argmax(y, axis=1)
    elif len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    return x, y


def load_mat_office31_AlexNet(filename):
    data = loadmat(filename)
    x = data['feas']
    y = data['labels'][0]
    return x, y


def u2t(x):
    """Convert uint8 to [-1, 1] float
    """
    max_num = 50000
    if len(x) > max_num:
        y = np.empty_like(x, dtype='float32')
        for i in range(len(x) // max_num):
            y[i*max_num: (i+1)*max_num] = (x[i*max_num: (i+1)*max_num].astype('float32') / 255) * 2 - 1

        y[(i + 1) * max_num:] = (x[(i + 1) * max_num:].astype('float32') / 255) * 2 - 1
    else:
        y = (x.astype('float32') / 255) * 2 - 1
    return y


class DataLoader:
    def __init__(self, src_domain=['mnistm'], trg_domain=['mnist'], data_path='./data', data_format='mat',
                 shuffle_data=False, dataset_name='digits', cast_data=True):
        self.num_src_domain = len(src_domain.split(','))
        self.src_domain_name = src_domain
        self.trg_domain_name = trg_domain
        self.data_path = data_path
        self.data_format = data_format
        self.shuffle_data = shuffle_data
        self.dataset_name = dataset_name
        self.cast_data = cast_data

        self.src_train = {}
        self.trg_train = {}
        self.src_test = {}
        self.trg_test = {}

        print("Source domains", self.src_domain_name)
        print("Target domain", self.trg_domain_name)
        self._load_data_train()
        self._load_data_test()

        self.data_shape = self.src_train[0][1][0].shape
        self.num_domain = len(self.src_train.keys())
        self.num_class = self.src_train[0][2].shape[-1]

    def _load_data_train(self, tail_name="_train"):
        if not self.src_train:
            self.src_train = self._load_file(self.src_domain_name, tail_name, self.shuffle_data)
            self.trg_train = self._load_file(self.trg_domain_name, tail_name, self.shuffle_data)

    def _load_data_test(self, tail_name="_test"):
        if not self.src_test:
            self.src_test = self._load_file(self.src_domain_name, tail_name, self.shuffle_data)
            self.trg_test = self._load_file(self.trg_domain_name, tail_name, self.shuffle_data)

    def _load_file(self, name_file=[], tail_name="_train", shuffle_data=False):
        data_list = {}
        name_file = name_file.split(',')
        for idx, s_n in enumerate(name_file):
            file_path_train = os.path.join(self.data_path, '{}{}.{}'.format(s_n, tail_name, self.data_format))
            if os.path.isfile(file_path_train):
                if self.dataset_name == 'digits':
                    x_train, y_train = load_mat_file_single_label(file_path_train)
                elif self.dataset_name == 'office31_AlexNet_feat':
                    x_train, y_train = load_mat_office31_AlexNet(file_path_train)
                if shuffle_data:
                    x_train, y_train = self.shuffle(x_train, y_train)
                if 'mnist32_60_10' not in s_n and self.cast_data:
                    x_train = u2t(x_train)
                data_list.update({idx: [s_n, x_train, to_categorical(y_train)]})
            else:
                raise('File not found!')
        return data_list

    def shuffle(self, x, y=None):
        np.random.seed(random_seed())
        idx_train = np.random.permutation(x.shape[0])
        x = x[idx_train]
        if y is not None:
            y = y[idx_train]
        return x, y

    def onehot2scalar(self, onehot_vectors, axis=1):
        return np.argmax(onehot_vectors, axis=axis)
