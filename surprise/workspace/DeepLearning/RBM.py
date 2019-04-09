# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:51:37 2019

@author: Ryan
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


class RBM(object):
    def __init__(self, visibleDimensions, epochs=20, hiddenDimensions=50)