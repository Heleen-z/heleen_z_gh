"""
自定义Instances类，支持翻滚属性
"""

import torch
import numpy as np
from ultralytics.utils.instance import Instances


class CustomInstances(Instances):
    """
    自定义Instances类，添加翻滚相关属性
    
    新增属性:
        roll (Tensor): 翻滚状态 [N], 0=stable, 1=suspect, 2=roll
        roll_conf (Tensor): 翻滚置信度 [N]
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._roll = None
        self._roll_conf = None
    
    @property
    def roll(self):
        """翻滚状态"""
        return self._roll
    
    @roll.setter
    def roll(self, value):
        self._roll = value
    
    @property
    def roll_conf(self):
        """翻滚置信度"""
        return self._roll_conf
    
    @roll_conf.setter
    def roll_conf(self, value):
        self._roll_conf = value
