#!/usr/bin/env python

#

# --- imports -----------------------------------------------------------------

import os
import time
import cv2
import numpy as np

def to_one_hot(labels):
    """
    Convert labels to one-hot
    :param labels:
    :return:
    """
    mx = max(labels)
    y = []
    for i in range(len(labels)):
        f = [0] * (mx + 1)
        f[labels[i]] = 1
        y.append(f)
    return y

def one_hot_encode(labels):
    """
    Convert labels to one-hot
    :param labels:
    :return:
    """
    if not isinstance(labels[0], list):
        mx = max(labels)
        y = []
        for i in range(len(labels)):
            f = [0] * (mx + 1)
            f[labels[i]] = 1
            y.append(f)
        return y
    else:
        return labels


def one_to_onehot(label, max_label):
    y = [0 for i in range(max_label)]
    y[label] = 1
    return y



def path_walk(path):
    path_list = []
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.lower().endswith(tuple(['.jpg', '.jpeg', '.png'])):
                path_list.append(os.path.abspath(os.path.join(dirpath, f)))
    return path_list


def log_loss_accuracy(accuracy, accuracy_type, task_type, num_classes):
    if isinstance(accuracy_type, list) and (len(accuracy_type) >= 2):
        acc_str = ''
        for k in range(len(accuracy_type)):
            acc_str += '{:s} '.format(accuracy_type[k])
            for i in range(num_classes):
                acc_str += 'lbl_{:d}: {:.3f} '.format(i, accuracy[k][i])
    else:
        if isinstance(accuracy_type, list):
            accuracy_type = accuracy_type[0]
        acc_str = '{:s}'.format(accuracy_type)
        if task_type == 'classification':
            acc_str += 'lbl: {:.3f} '.format(accuracy)
        elif task_type == 'reconstruction':
            acc_str += 'lbl: {:.3f} '.format(accuracy)
        else:
            raise ValueError('no task exists')
            # for i in range(num_classes):
            #     acc_str += 'lbl_{:d}: {:.3f} '.format(i, accuracy[i])

    return acc_str


def chunk_split(data_list, splits):
    avg = len(data_list) / float(splits)
    out = []
    last = 0.0

    while last < len(data_list):
        out.append(data_list[int(last):int(last + avg)])
        last += avg

    return out


