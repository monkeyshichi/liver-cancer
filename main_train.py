#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by Alan Chen on 2019/3/6

"""

import os

import cv2
import keras.backend.tensorflow_backend as KTF
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from keras.applications.densenet import preprocess_input
from tqdm import tqdm

from data_loaders.cancer_dl import CancerDL
from infers.cancer_infer import CancerInfer
from models.cancer_model import CancerModel
from trainers.cancer_trainer import CancerTrainer
from utils.config_utils import process_config

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)


def main_train():
    """
    训练模型

    :return:
    """
    print('[INFO] 解析配置...')

    model_config = process_config('configs/cancer_config.json')

    np.random.seed(47)  # 固定随机数

    print('[INFO] 构造网络...')
    model = CancerModel(config=model_config)

    print('[INFO] 加载数据...')
    dl = CancerDL(config=model_config)

    print('[INFO] 训练网络...')
    trainer = CancerTrainer(
        model=model.model,
        data=[dl.get_train_data(), dl.get_test_data()],
        config=model_config)
    trainer.train_generator()

    # load from saved npy
    # x3=np.load('x3.npy')
    # y3=np.load('y3.npy')
    # X_train, X_test, train_y, val_y = model_selection.train_test_split(x3, y3, random_state=2019, stratify=y3,
    #                                                                              test_size=0.2)
    # train_y = to_categorical(train_y)
    # val_y = to_categorical(val_y)
    # trainer = CancerTrainer(
    #     model=model.model,
    #     data=[(X_train,train_y), (X_test,val_y)],
    #     config=model_config)
    # trainer.train_generator()
    # print('[INFO] 训练完成...')


def read_ct_3d(ct_folder_path, config):
    slice_imgs = []
    for slice_name in sorted(os.listdir(ct_folder_path)):
        slice_folder_path = os.path.join(ct_folder_path, slice_name)
        slice_dicom = pydicom.read_file(slice_folder_path)
        slice_img = slice_dicom.pixel_array
        slice_img = cv2.resize(slice_img, (config.img_width, config.img_height))
        slice_imgs.append(slice_img)

    return np.stack(slice_imgs).astype(np.float32)


def get_data_batch(ct_3d_img):
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


SUBMIT_PATH = "submit.csv"


def calc_features(folder_path, train_label_path):
    df = pd.read_csv(train_label_path)
    model_config = process_config('configs/cancer_config.json')
    infer = CancerInfer("weights-best-all", model_config)
    cnt = 0
    for ct_folder_name in tqdm(os.listdir(folder_path)):
        ct_folder_path = os.path.join(folder_path, ct_folder_name)
        ct_3d_img = read_ct_3d(ct_folder_path, model_config)
        batch = get_data_batch(ct_3d_img)
        batch = np.expand_dims(batch, axis=0)
        x = preprocess_input(batch.astype(np.float32))
        # print("pre:",infer.predict(x))
        # cnt+=1
        # if cnt==50:
        #     break
        df['ret'][df.id == ct_folder_name] = np.argmax(infer.predict(x), axis=1)
    df.to_csv(SUBMIT_PATH, index=False)
    print(df.head())
    return df


def test_main():
    print('[INFO] 预测数据...')
    calc_features('../test_dataset', '../submit_example.csv')

    print('[INFO] 预测完成...')


if __name__ == '__main__':
    main_train()
    test_main()
