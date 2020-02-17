#!/usr/bin/env python

# --- imports -----------------------------------------------------------------
from network.TrainRunner import TrainRunner
from utils.code_computations import *



def training_procedure(experiment_id):
    training = TrainRunner(experiment_id=experiment_id)
    training.start_training()



if __name__ == "__main__":

    experiment_id = 'AutoEncoder_3l_lr0001_bs_256'
    training_procedure(experiment_id)

    # define parameters for further computations
    filepath = r'experiments/info_logs/AutoEncoder_3l_MNIST_on_two_const_lr_gradient/[gradient]_[0.1]_[256]'
    plotpath = r'experiments/plot_logs/AutoEncoder_3l_MNIST_on_two_const_lr_gradient/[gradient]_[0.1]_[256]'
    epoch_list = [0, 4, 9]
    fc_layers = [128, 64, 32]
    num_classes = 2
    num_epochs = 10
    mode_list = ['valid', 'train']

    # compute amount of codes per class
    uniqueness_count(filepath, 'train', 3, num_epochs, num_classes)
    uniqueness_count(filepath, 'valid', 3, num_epochs, num_classes)

    # plot the umap for each FC layer
    plot_umap(filepath, plotpath, mode_list, epoch_list, num_classes, fc_layers)

