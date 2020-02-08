#!/usr/bin/env python


# --- imports -----------------------------------------------------------------

from utils.metric import *
import tensorflow as tf
import numpy as np


class NetworkBase:
    """

    """

    def __init__(self, network_type, loss, accuracy, lr, training, optimizer=None, nonlin=None, num_filters=None, num_classes=None,
                 dropout=None, num_steps=None):
        """
        Construtcor of NetworkBase Class
        :param loss:        loss function
        :param lr:          learning rate
        :param training:    is training True/False
        :param optimizer:   optimizer
        :param nonlin:      nonliniearity
        :param upconv:      upconvolution method
        :param num_filters: number of filters
        :param num_classes: number of classes/labels
        :param dropout:     dropout ratio
        """
        self.loss_f = self._pick_loss_func(loss)
        self.is_training = training
        self.learning_rate = lr
        self.optimizer = optimizer
        self.network_type = network_type
        if optimizer:
            self.optimizer_f = self._pick_optimizer_func(optimizer)
        if nonlin:
            self.nonlin_f = self._pick_nonlin_func(nonlin)
        if num_filters:
            self.num_filters = num_filters
        if num_classes:
            self.num_classes = num_classes
        if dropout:
            self.dropout = dropout
        if num_steps:
            self.num_steps = num_steps

        self.weights_layer = 0
        self.biases_layer = 0
        self.bn_layer = 0

        self.trainable_layers = 'all'


        self.accuracy_f_list = self._pick_accuracy_function(accuracy)

    def _loss_function(self, y_pred, y_true):
        """
        Returns the loss
        :param y_pred:  prediction
        :param y_true:  ground truth
        :return:        loss
        """
        self.loss = self.loss_f(y_pred=y_pred, y_true=y_true)
        return self.loss

    def _accuracy_function(self, y_pred, y_true, net_type, loss_type):
        """
        Returns the accuracy
        :param y_pred:
        :param y_true:
        :return:
        """

        if loss_type == 'sigmoid':
            y_pred = tf.nn.sigmoid(y_pred)
        if loss_type == 'softmax':
            y_pred = tf.nn.softmax(y_pred)
        if net_type == 'classification':
            return self.accuracy_f_list(y_pred=y_pred, y_true=y_true)
        elif net_type == 'segmentation':
            return self.accuracy_f_list(y_pred=y_pred, y_true=y_true)
        elif net_type == 'reconstruction':
            return self.accuracy_f_list(y_pred=y_pred, y_true=y_true)
        else:
            raise ValueError('Unexpected network task %s' % net_type)

    def _optimizer_function(self, global_step=None, net_param=None):
        """
        Return the optimizer function
        :param global_step: current global step
        :return:            optimizer
        """

        if self.optimizer == 'momentum':
            return self.optimizer_f(self.learning_rate, momentum=0.9).minimize(self.loss, global_step=global_step)

        if global_step:
            return self.optimizer_f(self.learning_rate).minimize(self.loss, global_step=global_step)
        else:
            return self.optimizer_f(net_param, self.learning_rate)

    @staticmethod
    def _pick_nonlin_func(key):
        """
        Select a nonlinearity/activation function
        :param key: nonliniearity identifier
        :return:    nonliniearity/activation function
        """
        if key == 'elu':
            return tf.nn.elu
        elif key == 'relu':
            return tf.nn.relu
        elif key == 'lrelu':
            return tf.nn.leaky_relu
        elif key == 'tanh':
            return tf.nn.tanh
        else:
            raise ValueError('Unexpected nonlinearity function %s' % key)


    @staticmethod
    def _pick_loss_func(key):
        """
        Select loss function
        :param key: loss function identifier
        :return:    loss function
        """

        if key == 'softmax':
            return softmax_tf
        if key == 'sigmoid':
            return sigmoid_tf
        if key == 'margin':
            return margin_tf
        if key == 'mse':
            return mse_loss
        if key == 'mse_loss':
            return mse_loss_tf
        else:
            raise ValueError('Unexpected metric function %s' % key)


    @staticmethod
    def _pick_accuracy_function(key):
        if key == 'IoU':
            return IoU_tf
        elif key == 'dice_sorensen':
            return dice_sorensen_tf
        elif key == 'dice_jaccard':
            return dice_jaccard_tf
        elif key == 'mse':
            return mse_accuracy
        elif key == 'hinge':
            return hinge_tf
        elif key == 'percent':
            return percentage_tf
        else:
            raise ValueError('Unexpected metric function %s' % key)



    @staticmethod
    def _pick_optimizer_func(key):

        if key == 'adam':
            return tf.train.AdamOptimizer
        if key == 'momentum':
            return tf.train.MomentumOptimizer
        if key == 'gradient':
            return tf.train.GradientDescentOptimizer
        if key == 'proximalgrad':
            return tf.train.ProximalGradientDescentOptimizer
        else:
            raise ValueError('Unexpected optimizer function %s' % key)



    @staticmethod
    def _conv_bn_layer_tf(input_layer, n_filters, filter_scale=1, filter_size=3, is_training=True, nonlin_f=None,
                       padding='same', name='s_conv_bn', name_postfix='1_1'):
        """
        Convolution layer with batch normalization
        :param input_layer:     input layer
        :param n_filters:       number of filters
        :param filter_scale:    filter scale -> n_filters = n_filters * filter_scale
        :param filter_size:     filter size
        :param is_training:     training True/False
        :param nonlin_f:        nonlinearity/activation function, if None linear/no activation is used
        :param padding:         padding: valid or same
        :param name:            layer name
        :param name_postfix:    layer name postfix
        :return: conv, batch_norm, nonlin - convolution, batch normalization and nonlinearity/activation layer
        """
        with tf.name_scope(name + name_postfix):
            conv = tf.layers.conv2d(input_layer, filters=filter_scale * n_filters, kernel_size=filter_size,
                                    activation=None, padding=padding, name='conv_' + name_postfix,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1234),
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.1))
                                   # bias_initializer=tf.zeros_initializer())

            batch_norm = tf.layers.batch_normalization(conv, training=is_training, fused=False, name='batch_' + name_postfix)
            nonlin = nonlin_f(batch_norm, name='activation_' + name_postfix)
        return conv, batch_norm, nonlin

    @staticmethod
    def _conv_nonlin_layer(input_layer, n_filters, filter_scale=1, filter_size=3, nonlin_f=None, name_postfix='1_1'):
        """
        Convolution layer with batch normalization
        :param input_layer:     input layer
        :param n_filters:       number of filters
        :param filter_scale:    filter scale -> n_filters = n_filters * filter_scale
        :param filter_size:     filter size
        :param is_training:     training True/False
        :param nonlin_f:        nonlinearity/activation function, if None linear/no activation is used
        :param padding:         padding: valid or same
        :param name:            layer name
        :param name_postfix:    layer name postfix
        :return: conv, batch_norm, nonlin - convolution, batch normalization and nonlinearity/activation layer
        """
        with tf.name_scope('s_conv_nonlin' + name_postfix):
            conv = tf.layers.conv2d(input_layer, filters=filter_scale * n_filters, kernel_size=filter_size,
                                    activation=None, padding='same', name='conv_' + name_postfix,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1234),
                                    bias_initializer=tf.zeros_initializer())
            if nonlin_f:
                nonlin = nonlin_f(conv, name='activation_' + name_postfix)
            else:
                nonlin = None
        return conv, nonlin

    @staticmethod
    def _pool_layer(net, pool_size, stride_size, type='max', name='pooling'):
        """
        Pooling layer
        :param net:             network layer
        :param pool_size:       pooling size
        :param stride_size:     stride size
        :param type:            pooling type: max, average
        :param padding:         padding: valid or same
        :param name:            layer name
        :return:
        """
        pre_pool = net
        if type == 'max':
            net = tf.nn.max_pool(net, ksize=pool_size, strides=stride_size, padding='SAME', name=name)
        if type == 'aver':
            net = tf.nn.avg_pool(net, ksize=pool_size, strides=stride_size, padding='SAME', name=name)
        return pre_pool, net

    def return_accuracy(self, y_pred, y_true, net_type=None, loss_type=None):
        """
        Returns the prediction accuracy
        :param y_pred:  prediction
        :param y_true:  ground truth
        :return:        accuracy
        """
        return self._accuracy_function(y_pred=y_pred, y_true=y_true, net_type=net_type, loss_type=loss_type)

    def return_loss(self, y_pred, y_true):
        """
        Returns the loss
        :param y_pred:  prediction
        :param y_true:  ground truth
        :return:        loss
        """
        return self._loss_function(y_pred=y_pred, y_true=y_true)

    def return_optimizer(self, global_step=None, net_param=None):
        """
        Returns the optimizer function
        :param global_step: current global step
        :return:            optimizer
        """
        return self._optimizer_function(global_step, net_param)

    def return_nets(self):
        """
        Returns the network (only the convolutional layers)
        :return:    network
        """
        return self.nets

    @staticmethod
    def weights_and_biases(a, b):
        w = tf.Variable(tf.truncated_normal(shape=[a, b], stddev=np.sqrt(2 / a)))
        b = tf.Variable(tf.zeros([b]))

        return w, b