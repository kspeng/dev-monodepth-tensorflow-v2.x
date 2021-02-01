'''
MODEL   :: Tensorflow Computer Vision Platform
DATE    :: 2020-03-10
FILE    :: resnet18.py 
'''

from __future__ import absolute_import, division, print_function

from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow import layers as tfl
from tensorflow.contrib.layers import variance_scaling_initializer
from nn.nn_kits import NnKits

class Resnet18pe(object):
    def __init__(self, is_training):
        self.nn = NnKits()

    def forward(self, input, prefix=''):
        with tf.variable_scope('encoder'):    
            conv1 = self.nn.conv(input, 64, 7, 2, \
                    normalizer_fn=slim.batch_norm, \
                    activation_fn=tf.nn.relu) # H/2  -   64D
            pool1 = self.nn.maxpool(conv1,          3) # H/4  -   64D
            conv2 = self.nn.res33blocke(pool1,    64,1) # H/4  -   64D
            conv3 = self.nn.res33blocke(conv2,     128) # H/8 -  128D
            conv4 = self.nn.res33blocke(conv3,     256) # H/16 -  256D
            self.enc_feat   = self.nn.res33blocke(conv4,     512) # H/32 -  512D

        with tf.variable_scope('skips'):          
            self.skip1 = conv1
            self.skip2 = conv2
            self.skip3 = conv3
            self.skip4 = conv4
