#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import keras
from keras import backend as K
import tensorflow as tf
from importlib import import_module

class TagKerasInference(object):

    def __init__(self, **kwargs):
        _input_size = kwargs.get('input_size',299)
        gpu_rate = kwargs.get('gpu_rate',0.49)
        self.input_size = (_input_size, _input_size)
        self.gpu_index = kwargs.get('gpu_index', '0')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_index
        self.sess = tf.Session(config=self.GPU_config(rate=gpu_rate))
        K.set_session(self.sess)
        self.net = self._create_model(**kwargs)
        self._load(**kwargs)


    def close(self):
        self.sess.close()


    def GPU_config(self, rate=0.95):
        gpuConfig = tf.ConfigProto()
        gpuConfig.allow_soft_placement = True
        gpuConfig.gpu_options.allow_growth = True
        gpuConfig.gpu_options.per_process_gpu_memory_fraction = rate
        return gpuConfig


    def _create_model(self, **kwargs):
        is_official = kwargs.get('is_official', True)
        module_name = kwargs.get('module_name', 'inception_v3')
        net_name = kwargs.get('net_name', 'InceptionV3')
        if is_official:
            net_source = "keras.applications."
        else:
            net_source = "nets."
        m = import_module(net_source + module_name)
        model = getattr(m, net_name)
        include_top = True
        weights = None
        classes = kwargs.get('num_classes',1000)
        net = model(weights=weights, include_top=include_top, classes=classes)
        return net


    def _load(self, **kwargs):
        model_name = kwargs.get('model_name', 'model.h5')
        model_filename = model_name
        self.net.load_weights(model_filename)


    def run(self, image_data, **kwargs):
        _image_data = self.image_preprocess(image_data)
        input = np.expand_dims(_image_data, axis=0)
        result = self.net.predict(input)
        return result.tolist()


    def image_preprocess(self, image_data):
        _image = cv2.resize(image_data, self.input_size)
        _image = _image[:,:,::-1]   # bgr2rgb
        _image = (_image*1.0 - 127) * 0.0078125 # 1/128
        _image = _image.astype(np.float32)
        return _image.copy()


if __name__ == "__main__":
    # # python3 inference.py --image test.jpg --official True --module inception_v3 --net InceptionV3 --model model.h5 --cls 1000 --size 299
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-image', "--image", type=str, help='Assign the image path.', default=None)
    parser.add_argument('-module', "--module", type=str, help='Assign the module name.', default=None)
    parser.add_argument('-net', "--net", type=str, help='Assign the net name.', default=None)
    parser.add_argument('-model', "--model", type=str, help='Assign the net name.', default=None)
    parser.add_argument('-official', "--official", type=bool, help='Assign the official module.', default=None)
    parser.add_argument('-cls', "--cls", type=int, help='Assign the classes number.', default=None)
    parser.add_argument('-size', "--size", type=int, help='Assign the input size.', default=None)
    args = parser.parse_args()
    if args.image is None or args.module is None or args.net is None or args.model is None\
            or args.size is None or args.cls is None:
        raise TypeError('input error')
    if not os.path.exists(args.model):
        raise TypeError('cannot find file of model')
    print('test:')
    filename = args.image
    is_official = args.official
    module_name = args.module
    net_name = args.net
    model_name = args.model
    input_size = args.size
    num_classes = args.cls
    image = cv2.imread(filename)
    if image is None:
        raise TypeError('image data is none')
    # tagInfer = TagKerasInference(weights=None, include_top=True, classes=self.config['num_classes'])
    tagInfer = TagKerasInference(is_official=is_official, module_name=module_name,net_name=net_name,
                                 num_classes=num_classes, model_name=model_name,
                                   input_size=input_size)
    result = tagInfer.run(image)
    print(result)
    print('done!')