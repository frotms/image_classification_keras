# coding=utf-8
import os
import math
from collections import OrderedDict
import numpy as np
from utils import utils
from trainers.base_model import BaseModel
from nets.net_interface import NetModule

class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.config = config
        self.interface = NetModule(self.config['model_module_name'], self.config['model_net_name'],
                                   self.config['official_source'])
        self.gpus = len(utils.gpus_str_to_list(self.config['gpu_id']))
        self.create_model()


    def create_model(self):
        self.net = self.interface.create_model(weights=None, include_top=True, classes=self.config['num_classes'])
        # # compile the model (should be done *after* setting layers to non-trainable)
        # # for layer in self.model.net.layers[:25]:
        # #     layer.trainable = False

        # load weights before multi-gpu-model
        self.load()
        if self.gpus > 1:
            from keras.utils import multi_gpu_model
            self.net = multi_gpu_model(self.net, gpus=self.gpus)


    def load(self):
        # train_mode: 0:from scratch, 1:finetuning, 2:update
        train_mode = self.config['train_mode']
        if train_mode == 'fromscratch':
            print('from scratch...')

        elif train_mode == 'finetune':
            self._load()
            print('finetuning...')

        elif train_mode == 'update':
            self._load_all_weights()
            print('updating...')

        else:
            ValueError('train_mode is error...')


    def _load(self):
        import h5py
        f = h5py.File(os.path.join(self.config['pretrained_path'], self.config['pretrained_file']))
        # get weights of .h5 file
        state_dict = OrderedDict()
        for layer, g in f.items():
            for p_name in g.keys():
                param = g[p_name]
                tmp = []
                for k_name in param.keys():
                    kk = param.get(k_name)[:]
                    tmp.append(kk)
                state_dict[p_name] = tmp
        # get net tensor name
        model_tensor = self.net.layers
        tensor_list = [i.name for i in model_tensor]

        print('weights loading ...')
        # find node and weights in same name and size between .h5 file and net
        for k, v in state_dict.items():
            if k in tensor_list:
                layer_weights = self.net.get_layer(k).get_weights()
                for idx, weights in enumerate(layer_weights):
                    if v[idx].shape == weights.shape:
                        self.net.get_layer(k).set_weights(v)
        print('loaded weights done!')


    def _load_all_weights(self):
        pretrained_model_path = os.path.join(self.config['pretrained_path'], self.config['pretrained_file'])
        print('weights loading ...')
        self.net.load_weights(pretrained_model_path)
        print('loaded weights done!')
