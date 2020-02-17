#!/usr/bin/env python

#

# --- imports -----------------------------------------------------------------
import os
import gc
import ujson
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns



from tqdm import tqdm

def uniqueness_count(filepath, mode, num_fc, num_epochs, num_classes):
    for i in range(num_classes):
        class_distr = {'class': i,
                        'true_class': [],
                        'codes_true': [],
                        'points_true': []}

        code_lists_true = []
        res_true = []
        for ind in tqdm(range(0, num_epochs), total=num_epochs, unit=' files', desc='Processing {} files'.format(mode),
                        disable=False):
            with open(os.path.join(filepath, mode, '{}_codes_{}.json'.format(mode, ind + 1)), 'r') as dump:
                data = ujson.load(dump)
            lst_true = []
            for item in data:
                if item["Target"] == i:
                    lst_true.append(item["Code"][:])
            code_lists_true_n = []
            for n in range(num_fc):
                code_lists_true_n.append([lst[n] for lst in lst_true])

            all_true = [np.unique(np.ascontiguousarray(np.array(arr_true)).view(np.dtype((np.void, np.array(arr_true).dtype.itemsize * np.array(arr_true).shape[1]))) if arr_true else (np.array([]), [0]), return_counts=True) for arr_true in code_lists_true_n]
            res_true_n = [(all_true[i][0].view(np.array(code_lists_true_n[i]).dtype).reshape(-1, np.array(code_lists_true_n[i]).shape[1]).tolist(), all_true[i][1].tolist()) if all_true[i][0][0].size != 0 else ([], 0) for i in range(len(code_lists_true_n))]


            res_true.append(res_true_n)
            code_lists_true.append(code_lists_true_n)

        class_distr['true_class'] = [[len(x) for x in code_lists_true[i]] for i in range(len(code_lists_true))]
        class_distr['codes_true'] = [[x for x, _ in res_true[i]] for i in range(len(res_true))]
        class_distr['points_true'] = [[x for _, x in res_true[i]] for i in range(len(res_true))]

        with open(os.path.join(filepath, mode, '{}_code_sum_class_{}.json'.format(mode, i)), 'w') as f:
            ujson.dump(class_distr, f)

        del class_distr
        del all_true
        del res_true
        del code_lists_true
        gc.collect()



def compute_hamming_full(codes, fcl):
    mt = np.array(codes)
    ham_matrix = (2 * np.inner(mt - 0.5, 0.5 - mt) + mt.shape[1] / 2)
    del mt
    gc.collect()
    return ham_matrix/fcl



def hamming_dist_interclass_all_step_by_step(filepath, mode, num_classes, ep, fc, fc_list):
    ep_all = []
    for cl in range(num_classes):
        print(cl)
        with open(os.path.join(filepath, mode, '{}_code_sum_class_{}.json'.format(mode, cl)), 'r') as f:
            dist_dict = ujson.load(f)

        ep_lists = [dist_dict['codes_true'][ep][fc]]
        del dist_dict
        ep_all.append(ep_lists)

    hamm = compute_hamming_full([k for j in range(num_classes) for k in ep_all[j][0]], fc_list[fc])
    lbl = [j for j in range(num_classes) for k in ep_all[j][0]]
    X_tsne = umap.UMAP(n_neighbors=2,  min_dist=0.1, init='random').fit_transform(hamm)
    del hamm
    gc.collect()

    return X_tsne, lbl


def plot_umap(filepath, plotpath, mode_list, epoch_list, num_classes, fc_layers):
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
                plt.savefig(os.path.join(plotpath,
                                         '{}/Autoencoder_CIFAR10_{}_fc{}({})_epoch{}.png'.format(mode, mode, i_fc,
                                                                                                     fc_layers[i_fc],
                                                                                                     epoch)), dpi=400)
                plt.clf()
                plt.close()
                del umap
                del lbl
                gc.collect()
                print('done')