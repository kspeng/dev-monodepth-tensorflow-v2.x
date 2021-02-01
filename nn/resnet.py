'''
MODEL   :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-27
FILE    :: resnet.py 
'''

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nn.nn_kits import NnKits


class Resnet(object):

    def __init__(self, aspp_type='aspp'):
        self.nn = NnKits()

    def forward(self, input):
        with tf.variable_scope('encoder'):
            self.conv1 = self.nn.conv(input, 			64, 7, 2) # H/2  -   64D
            self.pool1 = self.nn.maxpool(self.conv1,           3) # H/4  -   64D
            self.conv2 = self.nn.resblock(self.pool1,      64, 3) # H/8  -  256D
            self.conv3 = self.nn.resblock(self.conv2,     128, 4) # H/16 -  512D
            self.conv4 = self.nn.resblock(self.conv3,     256, 6) # H/32 - 1024D
            self.enc_feat 	= self.nn.resblock(self.conv4,512, 3) # H/64 - 2048D

        with tf.variable_scope('skips'):
            self.skip1 = self.conv1
            self.skip2 = self.pool1
            self.skip3 = self.conv2
            self.skip4 = self.conv3
            self.skip5 = self.conv4