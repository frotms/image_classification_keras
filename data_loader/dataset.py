# -*- coding: utf-8 -*-

import os
import cv2
import copy
import math
import numpy as np
import keras
# import torch
# from torch.autograd import Variable
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
from data_loader.data_processor import DataProcessor


class KerasDataset(keras.utils.Sequence):
    def __init__(self, txt, config, batch_size=1, shuffle=True,
                 is_train_set=True):
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        imgs = []
        with open(txt,'r') as f:
            for line in f:
                line = line.strip('\n\r').strip('\n').strip('\r')
                words = line.split(self.config['file_label_separator'])
                # single label here so we use int(words[1])
                imgs.append((words[0], int(words[1])))

        self.DataProcessor = DataProcessor(self.config)
        self.imgs = imgs
        self.is_train_set = is_train_set
        self.on_epoch_end()


    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        batch_data = [self.imgs[k] for k in indexes]
        # Generate data
        images, labels = self._data_generation(batch_data)

        return images, labels


    def __len__(self):
        # calculate batch number of each epoch
        return math.ceil(len(self.imgs) / float(self.batch_size))


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.imgs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def _data_generation(self, batch_data):
        # Initialization
        images, labels = [], []
        _root_dir = self.config['train_data_root_dir'] if self.is_train_set else self.config['val_data_root_dir']
        # Generate data
        for idx, (path, label) in enumerate(batch_data):
            # Store sample
            filename = os.path.join(_root_dir, path)
            image = self.self_defined_loader(filename)
            images.append(image)
            # Store class
            labels.append(label)

        return np.array(images), keras.utils.to_categorical(labels, num_classes=self.config['num_classes'])
        # return np.array(images), np.array(labels) # keras.utils.to_categorical(labels, num_classes=self.n_classes)


    def self_defined_loader(self, filename):
        image = self.DataProcessor.image_loader(filename)
        image = self.DataProcessor.image_resize(image)
        if self.is_train_set and self.config['data_aug']:
            image = self.DataProcessor.data_aug(image)
        image = self.DataProcessor.input_norm(image)
        return image


def get_data_loader(config):
    """
    
    :param config: 
    :return: 
    """
    train_data_file = config['train_data_file']
    test_data_file = config['val_data_file']
    batch_size = config['batch_size']
    shuffle = config['shuffle']

    if not os.path.isfile(train_data_file):
        raise ValueError('train_data_file is not existed')
    if not os.path.isfile(test_data_file):
        raise ValueError('val_data_file is not existed')

    train_loader = KerasDataset(txt=train_data_file, config=config,
                              batch_size=batch_size, shuffle=shuffle,
                              is_train_set=True)
    test_loader = KerasDataset(txt=test_data_file, config=config,
                              batch_size=batch_size, shuffle=False,
                              is_train_set=False)

    # train_data = PyTorchDataset(txt=train_data_file,config=config,
    #                        transform=transforms.ToTensor(), is_train_set=True)
    # test_data = PyTorchDataset(txt=test_data_file,config=config,
    #                             transform=transforms.ToTensor(), is_train_set=False)
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle,
    #                          num_workers=num_workers)
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
    #                          num_workers=num_workers)

    return train_loader, test_loader



