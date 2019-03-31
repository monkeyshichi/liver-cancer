# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by Alan on 2019/3/6

"""
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD

from bases.model_base import ModelBase
import densenet
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


class CancerModel(ModelBase):
    """
    Lenet模型
    """

    def __init__(self, config):
        super(CancerModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        # print("w:",self.config.img_width)
        img_input = Input(shape=(self.config.img_width, self.config.img_height, 3), name='main_input')

        # base_model = DenseNet121(input_tensor=img_input, input_shape=(self.config.img_width, self.config.img_height, 3),
        #                          weights=None, include_top=False)

        # base_model = densenet.DenseNet(input_tensor=img_input,
        #                                input_shape=(self.config.img_width, self.config.img_height, 3),
        #                                weights=None, include_top=False, bottleneck=True, reduction=0.5, depth=40,
        #                                growth_rate=12,classes=self.config.num_class)
        #
        # # add a global spatial average pooling layer
        # x = base_model.output
        # x = GlobalAveragePooling2D()(x) #121 need this
        x = Conv2D(filters=6,
                   kernel_size=5,
                   strides=1,
                   activation='relu', input_shape=(self.config.img_width, self.config.img_height, 1))(img_input)
        x = MaxPooling2D(pool_size=2, strides=2)(x)
        x = Conv2D(filters=16,
                   kernel_size=5,
                   strides=1,
                   activation='relu')(x)
        x = MaxPooling2D(pool_size=2, strides=2)(x)
        x = Flatten()(x)
        x = Dense(units=120, activation='relu')(x)
        x = Dense(units=84, activation='relu')(x)

        # and a logistic layer
        predictions = Dense(self.config.num_class, activation='softmax')(x)

        # this is the model we will train
        model = Model(input=[img_input], output=predictions)  # base_model.input wu []
        # model = Model([img_input], predictions)
        sgd = SGD(lr=self.config.lr, decay=self.config.decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        self.model = model
