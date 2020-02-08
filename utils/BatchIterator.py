#!/usr/bin/env python

# --- imports -----------------------------------------------------------------
import cv2
import numpy as np
import sys


class BatchIterator:

    def __init__(self, file_name, data_split, img_size, num_lbls, batch_size, task_type, is_training, augment_dict=None,
                 shuffle_key=False, colormap=None):
        self.file_name = file_name
        self.data_split = data_split
        self.task_type = task_type
        self.is_training = is_training
        self.augment_dict = augment_dict
        self._preprocess_data(img_size, num_lbls, batch_size)


        if shuffle_key and not self.is_training:
            self._shuffle()
        else:
            self.permutation_list = self.data_split

        if self.task_type == 'classification' or self.task_type == 'reconstruction':
            it = [self.permutation_list[i:i+self.batch_size] for i in range(0, len(self.permutation_list), self.batch_size)]
            self.iterator = iter(it)

        self.current = next(self.iterator)

    def _preprocess_data(self, img_size, num_lbls, batch_size):
        self.X_key = self.file_name['X_data'][:]
        self.y_key = self.file_name['y_data'][:]
        self.size = len(self.data_split)
        self.img_size = img_size
        self.num_lbls = num_lbls
        self.batch_size = batch_size
        self.num_minibatch = int(self.size / self.batch_size) + 1
        self.count = 0
        if self.is_training:
            self.max_lim = self.size
        else:
            self.max_lim = self.size

    def get_max_lim(self):
        return int(self.max_lim / self.batch_size) + 1

    def _shuffle(self):
        import random
        self.permutation_list = random.sample(self.data_split, len(self.data_split))

    def __iter__(self):
        if self.task_type == 'classification':
            while self.count < self.num_minibatch:
                try:
                    yield self._next_batch_classification()
                except StopIteration:
                    break
        if self.task_type == 'reconstruction':
            while self.count < self.num_minibatch:
                try:
                    yield self._next_batch_reconstruction()
                except StopIteration:
                    break


    def _next_batch_classification(self):
        for i in self.current:
            imgs = self.X_key[i]/255
            lbls = self.y_key[i]
            yield imgs, lbls
        self.count = self.count + 1
        self.current = next(self.iterator)

    def _next_batch_reconstruction(self):
        for i in self.current:
            imgs = self.X_key[i].reshape(-1)/255
            lbls = self.y_key[i]
            yield imgs, lbls
        self.count = self.count + 1
        self.current = next(self.iterator)
