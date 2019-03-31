# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by Alan on 2019/3/6
"""

import os

import cv2
import numpy as np
import pandas as pd
import pydicom
from keras.applications.densenet import preprocess_input
from keras.utils import to_categorical
from sklearn import model_selection
from tqdm import tqdm

from bases.data_loader_base import DataLoaderBase

# img_width, img_height, img_channel = 224, 224, 3
TRAIN_PATH = '../train_dataset'
TRAIN_PATH2 = '../train_dataset2'
TRAIN_LABEL_PATH = '../train_label.csv'
TRAIN_LABEL_PATH2 = '../train2_label.csv'


class CancerDL(DataLoaderBase):
    def __init__(self, config=None):
        super(CancerDL, self).__init__(config)
        # self.X_train, self.y_train = self.calc_features(TRAIN_PATH,TRAIN_LABEL_PATH)
        #
        # self.X_train2, self.y_train2 = self.calc_features(TRAIN_PATH2,TRAIN_LABEL_PATH2)
        #
        # x3 = np.concatenate((self.X_train, self.X_train2), axis=0)
        # y3 = np.concatenate((self.y_train, self.y_train2), axis=0)
        # print("shape3:", x3.shape, y3.shape)
        # np.save('x3.npy', x3)
        # np.save('y3.npy', y3)
        x3 = np.load('x3.npy')
        y3 = np.load('y3.npy')
        self.X_train, self.X_test, train_y, val_y = model_selection.train_test_split(x3, y3, random_state=2019,
                                                                                     stratify=y3, test_size=0.2)
        cnt0 = 0

        x_index = []

        for i in range(y3.shape[0]):  # train_y
            if y3[i] == 1:  # train_y
                x_index.append(i)
                continue
            if cnt0 < 3296:
                x_index.append(i)
                cnt0 += 1
        self.X_train = x3[x_index]  # X_train
        train_y = y3[x_index]  # train_y
        self.y_train = to_categorical(train_y)    #train_y
        self.y_test = to_categorical(val_y)
        self.total_x = x3
        self.total_y = to_categorical(y3)
        print("train y0:", np.sum(train_y == 0))
        print("train y1:", np.sum(train_y == 1))
        print("val y:", np.sum(val_y == 1))

        print("[INFO] X_train.shape: %s, y_train.shape: %s" \
              % (str(self.X_train.shape), str(self.y_train.shape)))
        print("[INFO] X_test.shape: %s, y_test.shape: %s" \
              % (str(self.X_test.shape), str(self.y_test.shape)))

    def preprocess_data(self, x):
        x = x.astype(np.float32)
        x /= 255
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        x = (x - mean) / std
        return x

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_all_data(self):
        return self.total_x, self.total_y

    def read_ct_3d(self, ct_folder_path):
        slice_imgs = []
        for slice_name in sorted(os.listdir(ct_folder_path)):
            slice_folder_path = os.path.join(ct_folder_path, slice_name)
            slice_dicom = pydicom.read_file(slice_folder_path)
            slice_img = slice_dicom.pixel_array
            slice_img = cv2.resize(slice_img, (self.config.img_width, self.config.img_height))
            slice_imgs.append(slice_img)

        return np.stack(slice_imgs).astype(np.float32)

    def get_data_batch(self, ct_3d_img):
        # combine 3 slices to make RGB images
        batch = []
        for i in range(0, ct_3d_img.shape[0] - 3, 3):

            rgb_img = []
            for j in range(3):
                rgb_img.append(ct_3d_img[i + j])

            rgb_img = np.array(rgb_img)
            # channel first to channel last
            rgb_img = np.transpose(rgb_img, (1, 2, 0))
            batch.append(rgb_img)

        return np.mean(np.array(batch), axis=0)

    def calc_features(self, folder_path, train_label_path):
        x = []
        y = []
        df = pd.read_csv(train_label_path)

        for ct_folder_name in tqdm(os.listdir(folder_path)):
            ct_folder_path = os.path.join(folder_path, ct_folder_name)
            ct_3d_img = self.read_ct_3d(ct_folder_path)
            # print('ct_3d_img:', ct_3d_img.shape)
            batch = self.get_data_batch(ct_3d_img)
            # print('batch1:', batch.shape)
            x.append(batch)
            temp = df[df.id == ct_folder_name]["ret"].values
            y.extend(temp)

        x = preprocess_input(np.stack(x).astype(np.float32))
        print('x:', x.shape)
        print("y:", np.array(y).shape)
        return x, np.array(y)
