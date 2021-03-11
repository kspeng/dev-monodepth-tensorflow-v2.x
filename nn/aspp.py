'''
MODEL   :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-27
FILE    :: aspp.py 
'''

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from nn.nn_kits import NnKits

class ASPP(object):

    def __init__(self, type='aspp'):
        self.nn = NnKits()
        self.enc = self.aspp

    # Atrous helper
    def aspp(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """

        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keepdims=True)

        image_features = tf.keras.layers.Conv2D(depth, 1)(image_features)

        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        atrous_pool_block_1 = tf.keras.layers.Conv2D(depth, 1)(inputs)

        atrous_pool_block_6 = tf.keras.layers.Conv2D(depth, 3, dilation_rate=6, padding='same')(inputs)

        atrous_pool_block_12 = tf.keras.layers.Conv2D(depth, 3, dilation_rate=12, padding='same')(inputs)

        atrous_pool_block_18 = tf.keras.layers.Conv2D(depth, 3, dilation_rate=18, padding='same')(inputs)

        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, atrous_pool_block_12, atrous_pool_block_18), axis=3)

        return net  
