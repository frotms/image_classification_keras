# coding=utf-8

import os
import tensorflow as tf


class Logger:
    """
    tensorflow summary for tensorboard
    """
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log", "train")
        self.validate_summary_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log", "val")
        if not os.path.exists(self.train_summary_dir):
            os.makedirs(self.train_summary_dir)
        if not os.path.exists(self.validate_summary_dir):
            os.makedirs(self.validate_summary_dir)
        self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir, self.sess.graph)
        self.validate_summary_writer = tf.summary.FileWriter(self.validate_summary_dir)


    # it can summarize scalars and images.
    def data_summarize(self, step, summarizer="train", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the validate one
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.train_summary_writer if summarizer == "train" else self.validate_summary_writer
        if summaries_dict is not None:
            summary = tf.Summary()
            for tag, value in summaries_dict.items():
                summary.value.add(tag=tag, simple_value=value)
            summary_writer.add_summary(summary, step)
            summary_writer.flush()
