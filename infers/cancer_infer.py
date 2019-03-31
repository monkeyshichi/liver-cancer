# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by Alan on 2019/3/6
"""

from keras.models import model_from_json

from bases.infer_base import InferBase


class CancerInfer(InferBase):
    def __init__(self, name, config=None):
        super(CancerInfer, self).__init__(config)
        self.model = self.load_model(name)

    def load_model(self, name):
        # load json and create model
        with open(self.config.cp_dir + "model.json") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        # load weights into new model
        model.load_weights(self.config.cp_dir + name + ".h5")
        return model

    def predict(self, data):
        return self.model.predict(data)
