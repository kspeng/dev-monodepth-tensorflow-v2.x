'''
MODEL   :: Tensorflow Computer Vision Platform
DATE    :: 2020-03-10
FILE    :: resnet18.py 
'''

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from nn.nn_kits import NnKits

class Resnet18(object):
    def __init__(self, method='baseline'):
        self.nn = NnKits()
        self.method = method

    def forward(self, input, prefix=''):
        method = self.method

        with tf.compat.v1.variable_scope('encoder'):         
            conv1 = self.nn.conv(input, 64, 7, 2, \
                    normalizer_fn=tf.keras.layers.BatchNormalization, \
                    activation_fn=tf.nn.relu)           # H/2  -   64D
            pool1 = self.nn.maxpool(conv1,          3)  # H/4  -   64D
            conv2 = self.nn.res33block(pool1,      64, method=method) # H/4  -   64D
            conv3 = self.nn.res33block(conv2,     128, method=method) # H/8  -  128D
            conv4 = self.nn.res33block(conv3,     256, method=method) # H/16 -  256D
            self.enc_feat   = conv4

        with tf.compat.v1.variable_scope('skips'):          
            self.skip1 = conv1
            self.skip2 = pool1
            self.skip3 = conv2
            self.skip4 = conv3
