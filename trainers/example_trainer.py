# coding=utf-8
import os
import time
import keras
import keras.backend as K
from trainers.base_trainer import BaseTrainer
from utils import utils

class ExampleTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, config, logger):
        super(ExampleTrainer, self).__init__(model, train_loader, val_loader, config, logger)
        self._init()

    def _init(self):
        self.create_optimization()
        self.get_loss()
        self.compute_accuracy()


    def train(self):
        self.model.net.compile(optimizer=self.optimizer, loss=self.losser, metrics=self.metrics)
        total_epoch_num = self.config['num_epochs']
        if self.config['evaluate_before_train']:
            self.evaluate_epoch()
        for cur_epoch in range(1, total_epoch_num + 1):
            epoch_start_time = time.time()
            self.cur_epoch = cur_epoch
            self.train_epoch()
            self.evaluate_epoch()

            # printer
            self.logger.log_printer.epoch_case_print(self.cur_epoch,
                                                     self.eval_train, self.eval_validate,
                                                     self.train_losses.avg, self.eval_losses,
                                                     time.time() - epoch_start_time)
            # save model
            self.model.save()
            # logger
            self.logger.write_info_to_logger(variable_dict={'epoch': self.cur_epoch, 'lr': self.learning_rate,
                                                            'train_acc': self.eval_train,
                                                            'validate_acc': self.eval_validate,
                                                            'train_avg_loss': self.train_losses.avg,
                                                            'validate_avg_loss': self.eval_losses,
                                                            'gpus_index': self.config['gpu_id'],
                                                            'save_name': os.path.join(self.config['save_path'],
                                                                                      self.config['save_name']),
                                                            'net_name': self.config['model_net_name']})
            self.logger.write()
            # summary
            if self.config['is_tensorboard']:
                self.logger.summarizer.data_summarize(self.cur_epoch, summarizer='train',
                                                      summaries_dict={'train_acc': self.eval_train,
                                                                      'train_avg_loss': self.train_losses.avg})
                self.logger.summarizer.data_summarize(self.cur_epoch, summarizer='validate',
                                                      summaries_dict={'validate_acc': self.eval_validate,
                                                                      'validate_avg_loss': self.eval_losses})

    def train_epoch(self):
        """
        training in a epoch
        :return: 
        """
        # adjust learning_rate
        self.learning_rate = self.adjust_learning_rate(self.cur_epoch)
        self.train_losses = utils.AverageMeter()
        self.eval_train_batch = utils.AverageMeter()
        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            self.cur_batch = batch_idx + 1
            self.train_step(batch_x, batch_y)
            self.eval_train = self.eval_train_batch.avg

            # printer in an interval way
            if self.cur_batch % 10 == 0:
                self.logger.log_printer.iter_case_print(self.cur_epoch, self.eval_train, self.eval_validate,
                                                    len(self.train_loader), self.cur_batch, self.train_losses.avg, self.learning_rate)
            # logger
            self.logger.write_info_to_logger(variable_dict=None)
            # summary
            if self.config['is_tensorboard']:
                self.logger.summarizer.data_summarize(batch_idx, summarizer="train",
                                                      summaries_dict={"lr": self.learning_rate,
                                                                      'train_loss': self.train_losses.avg})
            # end
            if self.cur_batch % len(self.train_loader) == 0:
                self.train_loader.on_epoch_end()
                break

        time.sleep(1)


    def train_step(self, images, labels):
        """
        training in a step
        :param images: 
        :param labels: 
        :return: 
        """
        train_on = self.model.net.train_on_batch(images, labels, class_weight=None, sample_weight=None)
        self.train_losses.update(train_on[0], 1)
        self.eval_train_batch.update(train_on[1], 1)


    def get_loss(self):
        """
        compute loss
        :param pred: 
        :param label: 
        :return: 
        """
        self.losser = keras.losses.binary_crossentropy


    def create_optimization(self):
        """
        optimizer
        :return: 
        """
        self.optimizer = keras.optimizers.Adam(lr=self.config['learning_rate'],
                                               beta_1=0.9, beta_2=0.999, epsilon=None,
                                               decay=0.0, amsgrad=False)

    def adjust_learning_rate(self, epoch):
        """
        decay learning rate
        :param epoch: the first epoch is 1
        :return: 
        """
        learning_rate = self.config['learning_rate'] * (self.config['learning_rate_decay'] ** ((epoch - 1) // self.config['learning_rate_decay_epoch']))
        K.set_value(self.optimizer.lr, learning_rate)
        return learning_rate

    def compute_accuracy(self):
        """
        compute top-n accuracy
        :param output: 
        :param target: 
        :param topk: 
        :return: 
        """
        # self.metrics = [keras.metrics.mae, keras.metrics.categorical_accuracy]  # ['mae', 'acc'] # top_k_categorical_accracy
        self.metrics = [keras.metrics.categorical_accuracy]


    def evaluate_epoch(self):
        """
        evaluating in a epoch
        :return: 
        """
        res = self.model.net.evaluate_generator(generator=self.val_loader, steps=None, max_queue_size=10,
                                    workers=self.config['workers'], use_multiprocessing=False, verbose=0)
        self.eval_losses, self.eval_validate = res[0], res[1]
