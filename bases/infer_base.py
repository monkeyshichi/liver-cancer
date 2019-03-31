# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by Alan on 2019/3/6
"""


class InferBase(object):
    """
    推断基类
    """

    def __init__(self, config):
        self.config = config  # 配置

    def load_model(self, name):
        """
        加载模型
        """
        raise NotImplementedError

    def predict(self, data):
        """
        预测结果
        """
        raise NotImplementedError
