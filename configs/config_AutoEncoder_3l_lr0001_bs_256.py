#!/usr/bin/env python


from configs.config import ConfigFlags


def load_config():
    config = ConfigFlags().return_flags()

    config.autotune = False

    config.net = 'AutoEncoder_3l'
    config.training_mode = True
    config.image_size = [28, 28, 1]

    config.train_data_file = 'datasets/MNIST_on_two/MNIST_on_two_resized(28x28).hdf5_train'
    config.valid_data_file = 'datasets/MNIST_on_two/MNIST_on_two_resized(28x28).hdf5_train'
    config.test_data_file = 'datasets/MNIST_on_two/MNIST_on_two_resized(28x28).hdf5_test'

    config.lr = 0.1
    config.batch_size = 256
    config.num_epochs = 10
    config.num_classes = 2
    config.class_labels = [i for i in range(10)]
    config.num_filters = 64
    config.nonlin = 'relu'

    config.lr_decay = 0.1
    config.ref_steps = 3
    config.ref_patience = 3
    config.loss = 'sigmoid'
    config.nonlin = 'relu'
    config.optimizer = 'gradient'
    config.lr_mode = 'const_lr'
    config.task_type = 'reconstruction'
    config.accuracy = 'mse'

    config.gpu_load = 0.9
    config.augmentation = {'flip_hor': False,
                           'flip_vert': False}
    config.data_split = 0.7
    config.long_summary = False
    config.normalize = False
    config.zero_center = False
    config.dropout = 0.4
    config.chpnt2load = ''
    config.experiment_path = None

    return config
