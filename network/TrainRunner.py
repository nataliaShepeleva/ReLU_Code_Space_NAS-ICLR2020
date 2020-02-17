#!/usr/bin/env python


# --- imports -----------------------------------------------------------------
import gc
import os
import time
import h5py
import json
import ujson
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm


from network.NetRunner import NetRunner
from utils.BatchIterator import BatchIterator
from utils.utils import log_loss_accuracy

PYTHONHASHSEED=1234

class TrainRunner(NetRunner):
    def __init__(self, args=None, experiment_id=None):
        super().__init__(args, experiment_id)

    def get_var(self, all_vars, name):
        for i in range(len(all_vars)):
            if all_vars[i].name.startswith(name):
                v = all_vars[i].eval()
                v[np.isnan(v)] = 0
                return v
        return None

    def start_training(self):
        """
        Start Neural Network training
        :return:
        """
        self._initialize_training()
        # return validation_scores

    def _initialize_training(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.build_tensorflow_pipeline()
        self._run_tensorflow_pipeline()


    def _run_tensorflow_pipeline(self):
        tf.reset_default_graph()
        random.seed(1234)
        np.random.seed(1234)
        tf.compat.v1.random.set_random_seed(1234)

        if self.gpu_load != 0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_load)
            config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True, allow_soft_placement=True)
            config.gpu_options.allow_growth = True
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            config = tf.ConfigProto(device_count={'CPU': 1})


        h5_file_train = h5py.File(self.train_data_file, 'r')
        h5_file_valid = h5py.File(self.valid_data_file, 'r')

        data_split = [i for i in range(0, self.data_size)]

        pl = random.sample(data_split, len(data_split))

        with tf.Session(graph=self.graph, config=config) as sess:
            global_step = tf.train.get_global_step(sess.graph)
            self.net_params.append(global_step)
            if self.long_summary:
                long_summ, self.gradcam_summ = self._initialize_long_summary()
                self.net_params.append(long_summ)
            short_summ = self._initialize_short_summary()
            self.summ_params.append(short_summ)

            train_summary_writer = tf.summary.FileWriter(os.path.join(self.tr_path, 'train'), sess.graph)
            valid_summary_writer = tf.summary.FileWriter(os.path.join(self.tr_path, 'valid'), sess.graph)

            if not os.path.exists(os.path.join(self.info_path, 'train')):
                os.makedirs(os.path.join(self.info_path, 'train'))
            if not os.path.exists(os.path.join(self.info_path, 'valid')):
                os.makedirs(os.path.join(self.info_path, 'valid'))

            if not os.path.exists(os.path.join(self.plot_path, 'train')):
                os.makedirs(os.path.join(self.plot_path, 'train'))
            if not os.path.exists(os.path.join(self.plot_path, 'valid')):
                os.makedirs(os.path.join(self.plot_path, 'valid'))

            saver = tf.train.Saver(save_relative_paths=True, max_to_keep=10)

            self.graph_op.run()
            learn_rate = self.lr
            prev_loss = np.inf
            total_recall_counter_train = 0
            total_recall_counter_valid = 0

            for epoch in range(1, self.num_epochs + 1):
                train_generator = BatchIterator(h5_file_train, pl, self.img_size, self.num_classes, self.batch_size,
                                                self.task_type, self.is_training)
                train_lim = train_generator.get_max_lim()

                valid_generator = BatchIterator(h5_file_valid, [i for i in range(h5_file_valid['y_data'].shape[0])], self.img_size, self.num_classes, self.batch_size,
                                                self.task_type, False)
                valid_lim = valid_generator.get_max_lim()


                epoch_loss_train = 0
                epoch_accur_train = 0
                epoch_duration_train = 0

                epoch_loss_valid = 0
                epoch_accur_valid = 0
                epoch_duration_valid = 0

                test_dict = []
                valid_dict = []


                for i in tqdm(train_generator, total=train_lim, unit=' steps', desc='Epoch {:d} train'.format(epoch), disable=False):
                    start_time = time.time()
                    total_recall_counter_train += 1

                    ff = list(i)
                    # with tf.device('/device:GPU:0'):
                    if self.task_type == 'classification':
                        sess.run(self.enqueue_op, feed_dict={
                            self._in_data: [np.reshape(f[0], [self.img_size[0], self.img_size[1], self.img_size[2]]) for
                                            f
                                            in ff],
                            self._gt_data: [f[1] for f in ff]})
                    elif self.task_type == 'reconstruction':
                        sess.run(self.enqueue_op, feed_dict={
                            self._in_data: [f[0] for f in ff],
                            self._gt_data: [f[0] for f in ff]})
                        # labels = [f[1] for f in ff]
                    else:
                        raise ValueError('Task not found')

                    return_session_params = sess.run(self.net_params,
                                                     feed_dict={self.training_mode: True,
                                                                self.learning_rate: learn_rate})

                    if self.long_summary and total_recall_counter_train % train_generator.get_max_lim() == 1:
                        train_summary_writer.add_summary(return_session_params[-1], epoch)
                        # add Grad-CAM to the summary

                    with tf.device('/device:GPU:0'):
                        train_step_loss = return_session_params[2]
                        train_step_accuracy = return_session_params[3]
                        train_step_duration = time.time() - start_time
                        epoch_loss_train += train_step_loss

                        epoch_accur_train += train_step_accuracy
                        epoch_duration_train += train_step_duration

                    if self.network_type == 'AutoEncoder_3l':
                        with tf.device('/device:GPU:0'):
                            fc_1 = np.reshape(return_session_params[6],
                                              [int(len(return_session_params[6]) / 128), 128]).astype('int')
                            fc_2 = np.reshape(return_session_params[7],
                                              [int(len(return_session_params[7]) / 64), 64]).astype('int')
                            fc_3 = np.reshape(return_session_params[8],
                                              [int(len(return_session_params[8]) / 32), 32]).astype('int')
                            fc_1_ = (1 * (fc_1 > 0)).tolist()
                            fc_2_ = (1 * (fc_2 > 0)).tolist()
                            fc_3_ = (1 * (fc_3 > 0)).tolist()
                            test_dict.extend([{'Target': int(np.argmax(ff[k][1])),
                                               'Prediction': int(np.argmax(return_session_params[9][k])),
                                               'Code': [fc_1_[k], fc_2_[k], fc_3_[k]]} for k in
                                              range(return_session_params[5].shape[0])])

                with open(os.path.join(os.path.join(self.info_path, 'train'), 'train_codes_{}.json'.format(epoch)), 'w') as outfile:
                    ujson.dump(test_dict, outfile)




                for j in tqdm(valid_generator, total=valid_lim, unit=' steps', desc='Epoch {:d} valid'.format(epoch), disable=False):
                    start_time = time.time()
                    total_recall_counter_valid += 1

                    ff = list(j)
                    # with tf.device('/device:GPU:0'):
                    if self.task_type == 'classification':
                        sess.run(self.enqueue_op, feed_dict={
                            self._in_data: [np.reshape(f[0], [self.img_size[0], self.img_size[1], self.img_size[2]]) for f
                                            in ff],
                            self._gt_data: [f[1] for f in ff]})
                    elif self.task_type == 'reconstruction':
                        sess.run(self.enqueue_op, feed_dict={
                            self._in_data: [f[0] for f in ff],
                            self._gt_data: [f[0] for f in ff]})
                        # labels = [f[1] for f in ff]
                    else:
                        raise ValueError('Task not found')

                    return_session_params_valid = sess.run(self.net_params[1::],
                                                           feed_dict={self.training_mode: False,
                                                                      self.learning_rate: learn_rate})

                    if self.long_summary and total_recall_counter_valid % valid_generator.get_max_lim() == 1:
                        valid_summary_writer.add_summary(return_session_params_valid[-1], epoch)
                        # add Grad-CAM to the summary

                    with tf.device('/device:GPU:0'):
                        valid_step_loss = return_session_params_valid[1]
                        valid_step_accuracy = return_session_params_valid[2]
                        valid_step_duration = time.time() - start_time
                        epoch_loss_valid += valid_step_loss

                        epoch_accur_valid += valid_step_accuracy
                        epoch_duration_valid += valid_step_duration

                    if self.network_type == 'AutoEncoder_3l':
                        with tf.device('/device:GPU:0'):
                            fc_1 = np.reshape(return_session_params_valid[5],
                                              [int(len(return_session_params_valid[5]) / 128), 128]).astype('int')
                            fc_2 = np.reshape(return_session_params_valid[6],
                                              [int(len(return_session_params_valid[6]) / 64), 64]).astype('int')
                            fc_3 = np.reshape(return_session_params_valid[7],
                                              [int(len(return_session_params_valid[7]) / 32), 32]).astype('int')
                            fc_1_ = (1 * (fc_1 > 0)).tolist()
                            fc_2_ = (1 * (fc_2 > 0)).tolist()
                            fc_3_ = (1 * (fc_3 > 0)).tolist()
                            valid_dict.extend([{'Target': int(np.argmax(ff[k][1])),
                                               'Prediction': int(np.argmax(return_session_params_valid[8][k])),
                                               'Code': [fc_1_[k], fc_2_[k], fc_3_[k]]} for k in
                                              range(return_session_params_valid[4].shape[0])])
                        all_vars = tf.global_variables()
                        fc_1_w = self.get_var(all_vars, 'enc_1/kernel')
                        fc_2_w = self.get_var(all_vars, 'enc_2/kernel')
                        fc_3_w = self.get_var(all_vars, 'enc_3/kernel')
                        fc_1_b = self.get_var(all_vars, 'enc_1/bias')
                        fc_2_b = self.get_var(all_vars, 'enc_2/bias')
                        fc_3_b = self.get_var(all_vars, 'enc_3/bias')
                        data_dict = {'Weights': [fc_1_w.tolist(), fc_2_w.tolist(), fc_3_w.tolist()],
                                     'Bias': [fc_1_b.tolist(), fc_2_b.tolist(), fc_3_b.tolist()]}


                with open(os.path.join(self.weight_path, 'weights_bias_epoch_{}.json'.format(epoch)),
                          'w') as outfile:
                    json.dump(data_dict, outfile)

                with open(os.path.join(os.path.join(self.info_path, 'valid'), 'valid_codes_{}.json'.format(epoch)),
                          'w') as outfile:
                    ujson.dump(valid_dict, outfile)

                with tf.device('/device:GPU:0'):
                    train_aver_loss = epoch_loss_train / train_lim
                    epoch_accur_train = epoch_accur_train / train_lim
                    epoch_acc_str_tr = log_loss_accuracy(epoch_accur_train, self.accuracy_type, self.task_type,
                                                         self.num_classes)

                    valid_aver_loss = epoch_loss_valid / valid_lim
                    epoch_accur_valid = epoch_accur_valid / valid_lim
                    epoch_acc_str_valid = log_loss_accuracy(epoch_accur_valid, self.accuracy_type, self.task_type,
                                                         self.num_classes)


                    ret_train_epoch = sess.run(self.summ_params, feed_dict={self.learning_rate_plot: learn_rate,
                                                                            self.loss_plot: train_aver_loss,
                                                                            self.accuracy_plot: epoch_accur_train})

                    ret_valid_epoch = sess.run(self.summ_params, feed_dict={self.learning_rate_plot: learn_rate,
                                                                            self.loss_plot: valid_aver_loss,
                                                                            self.accuracy_plot: epoch_accur_valid})

                    train_summary_writer.add_summary(ret_train_epoch[-1], epoch)
                    valid_summary_writer.add_summary(ret_valid_epoch[-1], epoch)

                if not os.path.exists(self.ckpnt_path):
                    os.makedirs(self.ckpnt_path)

                print('\nRESULTS: epoch {:d} lr = {:.5f} train loss = {:.3f}, train accuracy : {:s} ({:.2f} sec) valid loss = {:.3f}, valid accuracy : {:s})'
                    .format(epoch, learn_rate, train_aver_loss, epoch_acc_str_tr, epoch_duration_train, valid_aver_loss, epoch_acc_str_valid, ))
                saver.save(sess, "{}/model.ckpt".format(self.ckpnt_path), global_step=epoch)



                gc.collect()

        h5_file_train.close()
        h5_file_valid.close()
