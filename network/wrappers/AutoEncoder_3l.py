#!/usr/bin/env python


# --- imports -----------------------------------------------------------------

import tensorflow as tf

from network.wrappers.NetworkBase import NetworkBase


class AutoEncoder_3l(NetworkBase):
    def __init__(self, network_type, loss, accuracy, lr, training, num_filters, optimizer='gradient', nonlin='relu',
                 num_classes=100, dropout=0.25):
        """
        Convolutional Neural Network constructor
        :param loss:        used loss function
        :param lr:          learning rate
        :param training:    is training True/False
        :param num_filters: number of filters
        :param optimizer:   used optimizer
        :param nonlin:      used nonliniearity
        :param num_classes: number of classes/labels
        :param dropout:     dropout ratio
        """
        super().__init__(network_type=network_type, loss=loss, accuracy=accuracy, lr=lr, training=training,
                         num_filters=num_filters, optimizer=optimizer, nonlin=nonlin,
                         num_classes=num_classes, dropout=dropout)

    def build_net(self, X):
        with tf.name_scope('encoder'):
            enc_1 = tf.layers.dense(X, units=128, activation=self.nonlin_f, name='enc_1',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
                                   bias_initializer=tf.zeros_initializer())
            enc_2 = tf.layers.dense(enc_1, units=64, activation=self.nonlin_f, name='enc_2',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
                                   bias_initializer=tf.zeros_initializer())
            enc_3 = tf.layers.dense(enc_2, units=32, activation=self.nonlin_f, name='enc_3',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
                                   bias_initializer=tf.zeros_initializer())
            pred_autoenc = tf.layers.dense(enc_3, units=self.num_classes, activation=None, name='output',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
                                       bias_initializer=tf.zeros_initializer())
        with tf.name_scope('decoder'):

            dec_3 = tf.layers.dense(enc_3, units=128, activation=self.nonlin_f, name='dec_3',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
                                   bias_initializer=tf.zeros_initializer())
            dec_2 = tf.layers.dense(dec_3, units=256, activation=self.nonlin_f, name='dec_2',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
                                   bias_initializer=tf.zeros_initializer())
            dec_1 = tf.layers.dense(dec_2, units=X.shape[-1], activation=None, name='dec_1',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0),
                                   bias_initializer=tf.zeros_initializer())

        return dec_1, tf.reshape(enc_1, [-1]), tf.reshape(enc_2, [-1]), tf.reshape(enc_3, [-1]), pred_autoenc