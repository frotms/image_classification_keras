# coding=utf-8
import os
from importlib import import_module

class NetModule(object):
    def __init__(self, module_name, net_name, is_official=True,**kwargs):
        self.module_name = module_name
        self.net_name = net_name
        if is_official:
            net_source = "keras.applications."
        else:
            net_source = "nets."
        # self.m = import_module('nets.' + self.module_name)
        self.m = import_module(net_source + self.module_name)

    def create_model(self, **kwargs):
        """
        when use a pretrained model of imagenet, pretrained_model_num_classes is 1000
        :param kwargs: 
        :return: 
        """
        _model = getattr(self.m, self.net_name)
        model = _model(**kwargs)
        return model