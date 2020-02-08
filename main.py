#!/usr/bin/env python

# --- imports -----------------------------------------------------------------
import sys
import importlib
from network.TrainRunner import TrainRunner
from utils.code_computations import *


if __name__ == "__main__":


    experiment_id = 'AutoEncoder_3l_lr0001_bs_256'
    config = importlib.import_module('configs.config_' + experiment_id)
    args = config.load_config()
    training = TrainRunner(experiment_id=experiment_id)
    training.start_training()


    filepath = r'\experiemnts\info_logs\AutoEncoder_CIFAR10_2cl_const_lr_gradient\[gradient]_[0.1]_[256]'
    plot_path = r'\experiemnts\\plot_logs\AutoEncoder_CIFAR10_2cl_const_lr_gradient\[gradient]_[0.1]_[256]_ISO'
    epoch_list = [0, 49, 99, 149, 199, 249, 299, 349, 399]
    fc_layers = [128, 64, 32]
    num_classes = 2
    num_epochs = 400

    mode_list = ['valid', 'train']



    uniqueness_count(filepath, 'train', 3, num_epochs, num_classes)
    uniqueness_count(filepath, 'valid', 3, num_epochs, num_classes)


    for mode in mode_list:
        for epoch in epoch_list:
            for i_fc in range(len(fc_layers)):
                umap, lbl = hamming_dist_interclass_all_step_by_step(filepath, mode, num_classes, epoch,
                                                                                 i_fc, fc_layers)
                sns.set_style('white')
                sns.despine()
                palette = sns.color_palette("bright", num_classes)
                sns.scatterplot(umap[:, 0], umap[:, 1], hue=lbl, palette=palette, alpha=0.4)
                plt.legend(title='classes', loc='center right', bbox_to_anchor=(1.15, 0.5), ncol=1)
                plt.xticks([], [])
                plt.yticks([], [])
                plt.title('Autoencoder CIFAR10 2 classes {} fc-{}({}) epoch-{} "isomorphic" via UMAP'.format(mode, i_fc,
                                                                                                             fc_layers[
                                                                                                                 i_fc],
                                                                                                             epoch + 1))
                plt.tight_layout()
                plt.savefig(os.path.join(plot_path,
                                         '{}/Autoencoder_CIFAR10_{}_fc{}({})_epoch{}_ISO.png'.format(mode, mode, i_fc,
                                                                                                     fc_layers[i_fc],
                                                                                                     epoch)), dpi=400)
                plt.clf()
                plt.close()
                del umap
                del lbl
                gc.collect()
                print('done')
