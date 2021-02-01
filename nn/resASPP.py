'''
MODEL   :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-27
FILE    :: vggASPP.py 
'''

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nn.nn_kits import NnKits
from nn.aspp import ASPP

class ResASPP(object):

    def __init__(self, aspp_type='aspp'):
        self.nn = NnKits()
        self.aspp = ASPP(aspp_type)

    def forward(self, input):
        with tf.variable_scope('encoder'):
            conv1 = self.nn.conv(input, 64, 7, 2) # H/2  -   64D
            pool1 = self.nn.maxpool(conv1,           3) # H/4  -   64D
            conv2 = self.nn.resblock(pool1,      64, 3) # H/8  -   64D
            conv3 = self.nn.resblock(conv2,     128, 4) # H/16 -  128D
            pool3 = self.nn.maxpool(conv3,           3) # H/32 -  128D  
            self.enc_feat   = self.aspp.enc(pool3) # H/32

        with tf.variable_scope('skips'):          
            self.skip1 = conv1
            self.skip2 = pool1
            self.skip3 = conv2
            self.skip4 = conv3
