# image-classification-keras
This repo is designed for those who want to start their projects of image classification.  
It provides fast experiment setup and attempts to maximize the number of projects killed within the given time.
It includes a few Convolutional Neural Network modules.You can build your own dnn easily.  
## Requirements  
Python3 support only. Tested on CUDA9.0, cudnn7.  
* albumentations==0.1.2  
* easydict==1.8  
* imgaug==0.2.6  
* opencv-python==3.4.3.18  
* protobuf==3.6.1  
* scikit-image==0.14.0  
* h5py==2.8.0  
* Keras==2.2.4  
* Keras-Applications==1.0.6  
* Keras-Preprocessing==1.0.5  
* tensorboard==1.9.0  
* tensorflow-gpu==1.9.0  

## model
| net                     | inputsize |
|-------------------------|-----------|
| vggnet                  | 224       |
| resnet                  | 224       |
| squeezenet              | 224       |
| densenet                | 224       |
| inceptionV3             | 299       |
| inceptionV4             | 299       |
| inception-resnet-v2     | 299       |
| xception                | 299       |
| mobilenet               | 224       |
| mobilenetV2             | 224       |
| nasnet-a-large          | 331       |
| nasnet-mobile           | 224       |
| squeezenet              | 224       |
| shufflenet              | 224       |
| shufflenetV2            | 224       |
| ...                     | ...       |  
### pre-trained model  
you can download pretrain model with url in ($net-module.py)  
#### From [keras-applications](https://github.com/keras-team/keras-applications/tree/master/keras_applications/) package (official):  
- resnet50 (`ResNet50`)  
- densenet (`DenseNet121`, `DenseNet169`, `DenseNet201`)  
- inception_v3 (`InceptionV3`)  
- vgg16 (`VGG16`)  
- vgg19 (`VGG19`)  
- mobilenet (`MobileNet`) 
- mobilenet_v2 (`MobileNetV2`)  
- inception_resnet_v2 (`InceptionResNetV2`)  
- xception (`Xception`)  
- nasnet (`NASNet`)  
[download the official weights here](https://github.com/fchollet/deep-learning-models/releases/)  
#### From non-official package:
- [inception_v4](https://github.com/kentsommer/keras-inceptionV4) (`inception_v4`)  [weights](https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels.h5)  [weights_notop](https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5)  
- [squeezenet](https://github.com/rcmalli/keras-squeezenet) (`SqueezeNet`)  [weights](https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5)  [weights_notop](https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5)  
- [shufflenet](https://github.com/scheckmedia/keras-shufflenet) (`ShuffleNet`)  
- [shufflenetv2](https://github.com/opconty/keras-shufflenetV2) (`ShuffleNetV2`)  
## usage

### configuration

| configure                       | description                                                               |
|---------------------------------|---------------------------------------------------------------------------|
| model_module_name               | eg: vgg_module                                                            |
| model_net_name                  | net function name in module, eg:vgg16                                     |
| gpu_id                          | eg: single GPU: "0", multi-GPUs:"0,1,3,4,7"                                                           |
| official_source                 | use official net. eg: true for using official net， false for diy-net     |
| async_loading                   | make an asynchronous copy to the GPU                                      |
| is_tensorboard                  | if use tensorboard for visualization                                      |
| evaluate_before_train           | evaluate accuracy before training                                         |
| shuffle                         | shuffle your training data                                                |
| data_aug                        | augment your training data                                                |
| img_height                      | input height                                                              |
| img_width                       | input width                                                               |
| num_channels                    | input channel                                                             |
| num_classes                     | output number of classes                                                  |
| batch_size                      | train batch size                                                          |
| workers                         | number of workers when evaluating data                                    |
| learning_rate                   | learning rate                                                             |
| learning_rate_decay             | learning rate decat rate                                                  |
| learning_rate_decay_epoch       | learning rate decay per n-epoch                                           |
| train_mode                      | eg:  "fromscratch","finetune","update"                                    |
| file_label_separator            | separator between data-name and label. eg:"----"                          |
| pretrained_path                 | pretrain model path                                                       |
| pretrained_file                 | pretrain model name. eg:"alexnet-owt-4df8aa71.pth"                        |
| pretrained_model_num_classes    | output number of classes when pretrain model trained. eg:1000 in imagenet |
| save_path                       | model path when saving                                                    |
| save_name                       | model name when saving                                                    |
| train_data_root_dir             | training data root dir                                                    |
| val_data_root_dir               | testing data root dir                                                     |
| train_data_file                 | a txt filename which has training data and label list                     |
| val_data_file                   | a txt filename which has testing data and label list                      |

### Training
1.make your training &. testing data and label list with txt file:  
txt file with single label index eg:  
	apple.jpg----0  
	k.jpg----3  
	30.jpg----0  
	data/2.jpg----1  
	abc.jpg----1  
2.configuration  
3.train  
	python3 train.py  
### Inference
	python3 inference.py --image test.jpg --official True --module inception_v3 --net InceptionV3 --model model.h5 --cls 1000 --size 299  
### tensorboard
	tensorboard --logdir=./logs/   
logdir is log dir in your project dir 
### experiment  
![](https://i.imgur.com/BvvwFLU.jpg)  
top-5:  
passion flower: 99.99864101409912%  
anthurium: 4.492078744533501e-05%  
clematis: 4.086914202616754e-05%  
barbeton daisy: 3.762780238503183e-05%  
great masterwort: 3.0815127161076816e-05%  
## References
1.[https://keras.io/](https://keras.io/)  
2.[https://github.com/keras-team/keras-applications](https://github.com/keras-team/keras-applications)  
3.[https://github.com/keras-team/keras](https://github.com/keras-team/keras)  
5.[https://github.com/Ahmkel/Keras-Project-Template](https://github.com/Ahmkel/Keras-Project-Template)  
4.[https://www.tensorflow.org](https://www.tensorflow.org)  
5.[https://github.com/fchollet/keras-resources](https://github.com/fchollet/keras-resources)  
6.[https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)  
7.[http://www.robots.ox.ac.uk/~vgg/data/flowers/102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102)  
8.[https://github.com/fchollet/deep-learning-models](https://github.com/fchollet/deep-learning-models)  
