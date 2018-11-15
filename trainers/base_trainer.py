# coding=utf-8
from __future__ import print_function
import os
import time


class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_train = -1.
        self.eval_validate = -1.
        self.eval_losses = -1.

    def train(self):
        """
        implement the logic of the training phase
        """
        # for cur_epoch in range(1, total_epoch_num+1):
        #     epoch_start_time = time.time()
        #     self.cur_epoch = cur_epoch
        #     self.train_epoch()
        #     self.evaluate_epoch()
        raise NotImplementedError


    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call teh train step
        """
        raise NotImplementedError


    def train_step(self):
        """
        implement the logic of the train step
        """
        raise NotImplementedError


    def evaluate_epoch(self):
        """
        implement the logic of epoch:
        -loop ever the number of iteration in the config and call teh train step
        """
        raise NotImplementedError


    def evaluate_step(self):
        """
        implement the logic of the train step
        """
        raise NotImplementedError


    def get_loss(self):
        """
        implement the logic of model loss
        """
        raise NotImplementedError


    def create_optimization(self):
        """
        implement the logic of the optimization
        """
        raise NotImplementedError