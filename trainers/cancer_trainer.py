# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by Alan on 2019/3/6
"""
import os
import warnings

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_recall_fscore_support
from keras.models import model_from_json
from bases.trainer_base import TrainerBase
from utils.np_utils import prp_2_oh_array
from utils.clr_callback import CyclicLR
from keras.optimizers import SGD


class CancerTrainer(TrainerBase):
    def __init__(self, model, data, config):
        super(CancerTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.cp_dir,
                                      '%s.weights.{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp_name),
                monitor="val_loss",
                mode='min',
                save_best_only=True,
                save_weights_only=False,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.tb_dir,
                write_images=True,
                write_graph=True,
            )
        )

        # self.callbacks.append(FPRMetric())
        self.callbacks.append(FPRMetricDetail())

    def train(self):
        history = self.model.fit(
            self.data[0][0], self.data[0][1],
            epochs=self.config.num_epochs,
            verbose=2,
            batch_size=self.config.batch_size,
            validation_data=(self.data[1][0], self.data[1][1]),
            # validation_split=0.25,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])

    def scheduler(self, epoch):
        # 每隔100个epoch，学习率减小为原来的1/10
        if epoch == 89 or epoch == 139:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(self.model.optimizer.lr)

    def train_generator(self):
        img_generator = ImageDataGenerator()

        data_x = self.data[0][0]
        data_y = self.data[0][1]
        permutation = np.random.permutation(data_x.shape[0])
        data_x = data_x[permutation, :, :]
        data_y = data_y[permutation]

        train_batches = img_generator.flow(data_x, data_y, batch_size=self.config.batch_size, seed=1)
        valid_batches = img_generator.flow(self.data[1][0], self.data[1][1], batch_size=self.config.batch_size, seed=1)
        # train_crops = crop_generator(train_batches, 32)

        filepath = self.config.cp_dir + "weights-best-all.h5"  # improvement-{epoch:02d}-{val_acc:.2f}
        checkpoint = ModelCheckpoint(filepath, verbose=1, save_weights_only=True, save_best_only=True,
                                     monitor='val_acc')
        callbacks_list = [checkpoint]

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto') #LearningRateScheduler(self.scheduler)

        print("summary:", self.model.summary())
        if os.path.exists(self.config.cp_dir + "model.json"):
            with open(self.config.cp_dir + "model.json") as json_file:
                model_json = json_file.read()
            model = model_from_json(model_json)
            sgd = SGD(lr=self.config.lr, decay=self.config.decay, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            # load weights into new model
            model.load_weights(self.config.cp_dir + "weights-best-all.h5")
            self.model = model
        else:
            model_json = self.model.to_json()
            with open(self.config.cp_dir + "model.json", "w") as json_file:
                json_file.write(model_json)
        clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                       step_size=800.)
        callbacks_list.append(clr)
        history = self.model.fit_generator(train_batches,
                                           steps_per_epoch=len(data_x) // self.config.batch_size, workers=8,
                                           use_multiprocessing=True,
                                           validation_data=valid_batches,
                                           validation_steps=len(self.data[1][0]) // self.config.batch_size,
                                           epochs=self.config.num_epochs, callbacks=callbacks_list)
        # serialize model to YAML

        plot_model(self.model, to_file='model.png')
        filepath2 = self.config.cp_dir + "model-best-all2.h5"
        self.model.save_weights(filepath2)

        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('./acc.png')
        plt.clf()
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('./loss.png')


class FPRMetric(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            val_y, prd_y, average='macro')
        print(" — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f" % (f_score, precision, recall))


class FPRMetricDetail(Callback):
    """
    输出F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        prd_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        precision, recall, f_score, support = precision_recall_fscore_support(val_y, prd_y)

        for p, r, f, s in zip(precision, recall, f_score, support):
            print(" — val_f1: % 0.4f — val_pre: % 0.4f — val_rec % 0.4f - ins %s" % (f, p, r, s))
