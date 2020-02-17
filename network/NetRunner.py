#!/usr/bin/env python

# --- imports -----------------------------------------------------------------
import os
import h5py
import time
import importlib
import tensorflow as tf

from network.wrappers import AutoEncoder_3l


class NetRunner:
    """

    """
    def __init__(self, args=None, experiment_id=None):
        self.X = None
        self.y = None
        self.X_valid = None
        self.y_valid = None
        self.num_classes = None

        self._parse_config(args, experiment_id)

    def _parse_config(self, args, experiment_id):
        """
        Read parameters from config files
        :param args:
        :param experiment_id:
        :return:
        """
        if not args:
            if experiment_id:
                config = importlib.import_module('configs.config_' + experiment_id)
                args = config.load_config()
            else:
                raise ValueError('No arguments or configuration data given')
        # Mandatory parameters for all architectures
        self.network_type = args.net
        self.is_training = args.training_mode
        self.train_data_file = args.train_data_file
        self.valid_data_file = args.valid_data_file
        self.test_data_file = args.test_data_file
        self.checkpoint_dir = args.checkpoint_dir
        self.trainlog_dir = args.trainlog_dir
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.loss_type = args.loss
        self.accuracy_type = args.accuracy
        self.optimizer = args.optimizer
        self.dropout = args.dropout
        self.gpu_load = args.gpu_load
        self.num_filters = args.num_filters
        self.nonlin = args.nonlin
        self.loss_type = args.loss
        self.task_type = args.task_type
        self.long_summary = args.long_summary
        self.experiment_path = args.experiment_path
        self.chpnt2load = args.chpnt2load
        self.lr_mode = args.lr_mode

        if not self.is_training:
            self.class_labels = args.class_labels
        if args.image_size:
            self.img_size = args.image_size
        else:
            self.img_size = None
        if args.num_classes:
            self.num_classes = args.num_classes
        else:
            self.num_classes = None
        if args.augmentation:
            self.augmentation_dict = args.augmentation
        else:
            self.augmentation_dict = None
        if args.normalize:
            self.normalize = args.normalize
        else:
            self.normalize = None
        if args.zero_center:
            self.zero_center = args.zero_center
        else:
            self.zero_center = None


        self._initialize_data()

    def _initialize_data(self):
        import numpy
        numpy.random.seed(0)
        if self.is_training:
            h5_file = h5py.File(self.train_data_file, 'r')
            self.data_size = h5_file['y_data'].shape[0]

            self.timestamp = str(time.time())

            self.experiment_path = os.path.join(os.path.dirname(os.path.abspath('utils')), "experiments")
            if not os.path.exists(self.experiment_path):
                os.makedirs(self.experiment_path)
            self.info_path = os.path.join(self.experiment_path, "info_logs", '{}_{}_{}_{}'.format(self.network_type, os.path.split(os.path.split(self.train_data_file)[0])[1], self.lr_mode, self.optimizer), '[{}]_[{}]_[{}]'.format(self.optimizer, self.lr, self.batch_size))
            if not os.path.exists(self.info_path):
                os.makedirs(self.info_path)
            self.tr_path = os.path.join(self.experiment_path, "train_logs", '{}_{}_{}_{}'.format(self.network_type, os.path.split(os.path.split(self.train_data_file)[0])[1], self.lr_mode, self.optimizer), '[{}]_[{}]_[{}]'.format(self.optimizer, self.lr, self.batch_size))
            if not os.path.exists(self.tr_path):
                os.makedirs(self.tr_path)
            self.ckpnt_path = os.path.join(self.experiment_path, "ckpnt_logs", '{}_{}_{}_{}'.format(self.network_type, os.path.split(os.path.split(self.train_data_file)[0])[1], self.lr_mode, self.optimizer), '[{}]_[{}]_[{}]'.format(self.optimizer, self.lr, self.batch_size))
            if not os.path.exists(self.ckpnt_path):
                os.makedirs(self.ckpnt_path)
            self.weight_path = os.path.join(self.experiment_path, "weight_logs", '{}_{}_{}_{}'.format(self.network_type, os.path.split(os.path.split(self.train_data_file)[0])[1], self.lr_mode, self.optimizer), '[{}]_[{}]_[{}]'.format(self.optimizer, self.lr, self.batch_size))
            if not os.path.exists(self.weight_path):
                os.makedirs(self.weight_path)
            self.plot_path = os.path.join(self.experiment_path, "plot_logs", '{}_{}_{}_{}'.format(self.network_type, os.path.split(os.path.split(self.train_data_file)[0])[1], self.lr_mode, self.optimizer), '[{}]_[{}]_[{}]'.format(self.optimizer, self.lr, self.batch_size))
            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)
            h5_file.close()
        else:
            h5_file = h5py.File(self.test_data_file, 'r')
            self.inference_X = h5_file['X_data'][:]
            self.inference_y = h5_file['y_data'][:]
            h5_file.close()


    def build_tensorflow_pipeline(self):

        self.graph = tf.Graph()
        with self.graph.as_default():

            if self.task_type == 'classification':
                in_shape = [None, self.img_size[0], self.img_size[1], self.img_size[2]]
                gt_shape = [None, self.num_classes]
                self._in_data = tf.placeholder(tf.float32, shape=in_shape, name='Input_train')
                self._gt_data = tf.placeholder(tf.float32, shape=gt_shape, name='GT_train')
            elif self.task_type == 'reconstruction':
                in_shape = [None, self.img_size[0] * self.img_size[1] * self.img_size[2]]
                gt_shape = [None, self.img_size[0] * self.img_size[1] * self.img_size[2]]
                self._in_data = tf.placeholder(tf.float32, shape=in_shape, name='Input_train')
                self._gt_data = tf.placeholder(tf.float32, shape=gt_shape, name='GT_train')
            else:
                raise ValueError('No task exists')


            queue = tf.FIFOQueue(self.data_size * self.num_epochs, [tf.float32, tf.float32], name='queue')
            self.enqueue_op = queue.enqueue([self._in_data, self._gt_data])
            self.in_data, self.gt_data = queue.dequeue()
            self.in_data.set_shape(in_shape)
            self.gt_data.set_shape(gt_shape)

            self.learning_rate = tf.placeholder(tf.float32, name='Learning_rate')
            self.training_mode = tf.placeholder(tf.bool, name='Mode_train')

            self.epoch_loss = tf.placeholder(tf.float32, name='Epoch_loss')
            self.epoch_accuracy = tf.placeholder(tf.float32, name='Epoch_accuracy')
            self.loss_plot = tf.placeholder(tf.float32, name='Epoch_loss')
            self.learning_rate_plot = tf.placeholder(tf.float32, name='Epoch_learn')

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


            self.network = self._pick_model()


            if self.network_type == 'AutoEncoder_3l':
                self.pred_output, self.code_fc1, self.code_fc2, self.code_fc3, self.autoenc_label = self.network.build_net(self.in_data)

            self.global_step = tf.train.get_or_create_global_step()

            with tf.control_dependencies(self.update_ops):
                self.loss = self.network.return_loss(y_pred=self.pred_output, y_true=self.gt_data)
                self.train_op = self.network.return_optimizer(global_step=self.global_step)
                self.accuracy = self.network.return_accuracy(y_pred=self.pred_output, y_true=self.gt_data,
                                                             net_type=self.task_type, loss_type=self.loss_type)

            self.accuracy_plot = tf.placeholder(tf.float32, name='Epoch_accuracy')

            tf.add_to_collection('mode_train', self.training_mode)
            tf.add_to_collection('inputs_train', self.in_data)
            tf.add_to_collection('outputs_train', self.pred_output)
            # tf.add_to_collection('code_outputs',  self.code_outputs)
            tf.add_to_collection('learn_rate', self.learning_rate)

            if self.network_type == 'AutoEncoder_3l':
                tf.add_to_collection('code_fc1', self.code_fc1)
                tf.add_to_collection('code_fc2', self.code_fc2)
                tf.add_to_collection('code_fc3', self.code_fc3)
                tf.add_to_collection('pred_autoenc', self.autoenc_label)
                self.net_params = [self.train_op, self.training_mode, self.loss, self.accuracy,
                                   self.learning_rate, self.pred_output, self.code_fc1, self.code_fc2, self.code_fc3,
                                   self.autoenc_label]

            self.summ_params = [self.learning_rate_plot, self.loss_plot, self.accuracy_plot]

            self.graph_op = tf.global_variables_initializer()

    def _pick_model(self):
        """
        Pick a deep model specified by self.network_type string
        :return:
        """

        if self.network_type == 'AutoEncoder_3l':
            return AutoEncoder_3l.AutoEncoder_3l(self.network_type, self.loss_type, self.accuracy_type, self.learning_rate,
                                       training=self.is_training, num_filters=self.num_filters, nonlin=self.nonlin,
                                       num_classes=self.num_classes, optimizer=self.optimizer)

        else:
            raise ValueError('Architecture does not exist')


    def _initialize_short_summary(self):
        """
        Tensorboard scope initialization
        :return:
        """
        loss_sum = tf.summary.scalar('Loss_function', self.loss_plot)
        lr_summ = tf.summary.scalar("Learning_rate", self.learning_rate_plot)
        # more then one accuracy specified
        if isinstance(self.accuracy_type, list) and (len(self.accuracy_type) >= 2):
            acc_summ = tf.stack([tf.summary.scalar('{:s}_Accuracy_label_{:d}'.format(self.accuracy_type[k], i),
                                                   self.accuracy_plot[k][i]) for k in range(len(self.accuracy_type))
                                 for
                                 i in range(self.num_classes)])
        # only one accuracy specified
        else:
            if isinstance(self.accuracy_type, list):
                self.accuracy_type = self.accuracy_type[0]
            if self.task_type == 'classification':
                acc_summ = tf.summary.scalar('{:s}_accuracy'.format(self.accuracy_type), self.accuracy_plot)
            elif self.task_type == 'reconstruction':
                acc_summ = tf.summary.scalar('{:s}_accuracy'.format(self.accuracy_type), self.accuracy_plot)

        summary_op = tf.summary.merge([loss_sum, lr_summ, acc_summ])
        return summary_op

    def _initialize_long_summary(self):
        img_summ = tf.summary.image("Original_Image", self.in_data, self.batch_size)
        if self.task_type == 'classification':
            gt_summ = tf.summary.text('Input_labels',
                                      tf.as_string(
                                          tf.reshape(tf.argmax(self.gt_data, axis=1), [-1, self.batch_size])))
            pred_summ = tf.summary.text('Predicted_labels', tf.as_string(
                tf.reshape(tf.argmax(self.pred_output, axis=1), [-1, self.batch_size])))
        # Setup Grad-CAM summaries


        hist_summ = tf.stack(
            [tf.summary.histogram(tf.trainable_variables()[i].name[:-2] + '_train', tf.trainable_variables()[i]) for
             i
             in range(len(tf.trainable_variables()))])
        m_summs = [img_summ, gt_summ, pred_summ, hist_summ]
        summary_op = tf.summary.merge(m_summs)

        return summary_op, None