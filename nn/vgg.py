'''
MODEL   :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-27
FILE    :: vgg.py 
'''

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nn.nn_kits import NnKits

class Vgg(object):

    def __init__(self, aspp_type='aspp'):
        self.nn = NnKits()
        
    def forward(self, input):
        with tf.variable_scope('encoder'):
            self.conv1 = self.nn.conv_block(input,  32, 7) # H/2
            self.conv2 = self.nn.conv_block(self.conv1,       64, 5) # H/4
            self.conv3 = self.nn.conv_block(self.conv2,      128, 3) # H/8
            self.conv4 = self.nn.conv_block(self.conv3,      256, 3) # H/16
            self.conv5 = self.nn.conv_block(self.conv4,      512, 3) # H/32
            self.conv6 = self.nn.conv_block(self.conv5,      512, 3) # H/64
            self.enc_feat = self.nn.conv_block(self.conv6,   512, 3) # H/128                

        with tf.variable_scope('skips'):
            self.skip1 = self.conv1
            self.skip2 = self.conv2
            self.skip3 = self.conv3
            self.skip4 = self.conv4
            self.skip5 = self.conv5
            self.skip6 = self.conv6