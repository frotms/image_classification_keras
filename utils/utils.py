# coding=utf-8
from __future__ import print_function
import numpy as np
import time
import sys
import os


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def view_bar(num, total):
    """
    
    :param num: 
    :param total: 
    :return: 
    """
    rate = float(num + 1) / total
    rate_num = int(rate * 100)
    if num != total:
        r = '\r[%s%s]%d%%' % ("=" * rate_num, " " * (100 - rate_num), rate_num,)
    else:
        r = '\r[%s%s]%d%%' % ("=" * 100, " " * 0, 100,)
    sys.stdout.write(r)
    sys.stdout.flush()


def gpus_str_to_list(gpus_index_str):
    """
    convert gpu-environ str to list
    :param gpus_index_str: 
    :return: 
    """
    gpus_list = []
    _str = gpus_index_str.replace(' ','')
    _list = _str.split(',')
    for i, val in enumerate(_list):
        gpus_list.append(int(val))
    return gpus_list


def GPU_config(config):
    """
    configuration of gpu
    :return: 
    """
    gpuConfig = tf.ConfigProto()
    gpuConfig.allow_soft_placement = config['gpu_allow_soft_placement']
    gpuConfig.gpu_options.allow_growth = config['gpu_allow_growth']
    gpuConfig.gpu_options.per_process_gpu_memory_fraction = config['gpu_rate']
    gpuConfig.log_device_placement = config['gpu_log_device_placement']
    return gpuConfig