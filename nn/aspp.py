'''
MODEL   :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-27
FILE    :: aspp.py 
'''

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nn.nn_kits import NnKits

class ASPP(object):

    def __init__(self, type='aspp'):
        self.nn = NnKits()
        if type == 'aspp':
            self.enc = self.aspp
        elif type == 'asppr':
            self.enc = self.asppr    
        elif type == 'asppr2':
            self.enc = self.asppr2                     
        elif type == 'asppc':
            self.enc = self.asppc
        elif type == 'asppcv':
            self.enc = self.asppcv            
        elif type == 'asppcs':
            self.enc = self.asppcs
        elif type == 'asppcso':
            self.enc = self.asppcso            
        elif type == 'asppcsr':
            self.enc = self.asppcsr         
        elif type == 'asppcsr2':
            self.enc = self.asppcsr2          
        elif type == 'asppcsre2':
            self.enc = self.asppcsre2            
        elif type == 'asppcsre':
            self.enc = self.asppcsre            
        else:
            print('>>> ASPP type {} is not supported. <<<'.format(type))        

    def resAtrousConv(self, x, depth, kernel_size=1, rate=1, num_bolck=1, activation_fn=tf.nn.elu):
        shortcut = x
        for i in range(num_bolck-1):
            x = slim.conv2d(x, depth, [kernel_size, kernel_size], rate=rate, normalizer_fn=slim.batch_norm, activation_fn=activation_fn)
        x = slim.conv2d(x, depth, [kernel_size, kernel_size], rate=rate, normalizer_fn=slim.batch_norm, activation_fn=None) + shortcut
        return activation_fn(x)

    def resAtrousConve(self, x, depth, kernel_size=1, rate=1, num_bolck=1, activation_fn=tf.nn.elu):
        #shortcut = x
        shortcut = self.nn.conv(x, depth, 3, 1, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.elu)
        #shortcut = slim.conv2d(x, depth, [kernel_size, kernel_size], rate=rate, normalizer_fn=slim.batch_norm, activation_fn=activation_fn)
        for i in range(num_bolck-1):
            x = slim.conv2d(x, depth, [kernel_size, kernel_size], rate=rate, normalizer_fn=slim.batch_norm, activation_fn=activation_fn)
        x = slim.conv2d(x, depth, [kernel_size, kernel_size], rate=rate, normalizer_fn=slim.batch_norm, activation_fn=None) + shortcut
        return activation_fn(x)

    # Atrous helper
    def aspp(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """

        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth, [1, 1], activation_fn=None)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        atrous_pool_block_1 = slim.conv2d(inputs, depth, [1, 1], activation_fn=None)

        atrous_pool_block_6 = slim.conv2d(inputs, depth, [3, 3], rate=6, activation_fn=None)

        atrous_pool_block_12 = slim.conv2d(inputs, depth, [3, 3], rate=12, activation_fn=None)

        atrous_pool_block_18 = slim.conv2d(inputs, depth, [3, 3], rate=18, activation_fn=None)

        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, atrous_pool_block_12, atrous_pool_block_18), axis=3)

        return net  

    def asppr(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """
        num_bolck_ = 1
        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth, [1, 1], activation_fn=None)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        atrous_pool_block_1 = self.resAtrousConv(inputs, depth, kernel_size=1, rate=1, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        atrous_pool_block_6 = self.resAtrousConv(inputs, depth, kernel_size=3, rate=6, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        atrous_pool_block_12 = self.resAtrousConv(inputs, depth, kernel_size=3, rate=12, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        atrous_pool_block_18 = self.resAtrousConv(inputs, depth, kernel_size=3, rate=18, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, atrous_pool_block_12, atrous_pool_block_18), axis=3)

        return net          

    def asppr2(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """
        num_bolck_ = 2
        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth, [1, 1], activation_fn=None)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        atrous_pool_block_1 = self.resAtrousConv(inputs, depth, kernel_size=1, rate=1, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        atrous_pool_block_6 = self.resAtrousConv(inputs, depth, kernel_size=3, rate=6, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        atrous_pool_block_12 = self.resAtrousConv(inputs, depth, kernel_size=3, rate=12, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        atrous_pool_block_18 = self.resAtrousConv(inputs, depth, kernel_size=3, rate=18, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, atrous_pool_block_12, atrous_pool_block_18), axis=3)

        return net  


    def asppc(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """

        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth, [1, 1], activation_fn=None)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        inputs_feat = inputs # tf.concat([image_features, inputs], 3)
        atrous_pool_block_1 = slim.conv2d(inputs_feat, depth, [1, 1], activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_1], 3)
        atrous_pool_block_3 = slim.conv2d(inputs_feat, depth, [3, 3], rate=3, activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_3], 3)
        atrous_pool_block_6 = slim.conv2d(inputs_feat, depth, [3, 3], rate=6, activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_6], 3)
        atrous_pool_block_12 = slim.conv2d(inputs_feat, depth, [3, 3], rate=12, activation_fn=None)

        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_3, atrous_pool_block_6, atrous_pool_block_12), axis=3)

        return net  

    def asppcv(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """

        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth, [1, 1], activation_fn=None)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        inputs_feat = inputs # tf.concat([image_features, inputs], 3)
        atrous_pool_block_1 = slim.conv2d(inputs_feat, depth, [1, 1], activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_1], 3)
        atrous_pool_block_3 = slim.conv2d(inputs_feat, depth, [3, 3], rate=3, activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_3], 3)
        atrous_pool_block_6 = slim.conv2d(inputs_feat, depth, [3, 3], rate=6, activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_6], 3)
        atrous_pool_block_12 = slim.conv2d(inputs_feat, depth, [3, 3], rate=12, activation_fn=None)

        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_3, atrous_pool_block_6, atrous_pool_block_12), axis=3)

        aspp_feat = self.nn.conv(net, depth, 3, 1)

        return aspp_feat 

    def asppcs(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (3, 6, 12) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """

        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth/2, [1, 1], activation_fn=None)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        inputs_feat = tf.concat([image_features, inputs], 3)
        atrous_pool_block_1 = slim.conv2d(inputs_feat, depth/2, [1, 1], activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_1], 3)
        atrous_pool_block_3 = slim.conv2d(inputs_feat, depth/2, [3, 3], rate=3, activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_3], 3)
        atrous_pool_block_6 = slim.conv2d(inputs_feat, depth/2, [3, 3], rate=6, activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_6], 3)
        atrous_pool_block_12 = slim.conv2d(inputs_feat, depth/2, [3, 3], rate=12, activation_fn=None)

        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_3, atrous_pool_block_6, atrous_pool_block_12), axis=3)

        aspp_feat = self.nn.conv(net, depth, 3, 1)

        return aspp_feat   

    def asppcsr(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (3, 6, 12) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """
        bn=slim.batch_norm 
        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth/2, [1, 1], activation_fn=None)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        inputs_feat = tf.concat([image_features, inputs], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)
        atrous_pool_block_1 = slim.conv2d(x, depth/2, [1, 1], activation_fn=None) + x

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_1], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)        
        atrous_pool_block_3 = slim.conv2d(x, depth/2, [3, 3], rate=3, activation_fn=None) + x

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_3], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)        
        atrous_pool_block_6 = slim.conv2d(x, depth/2, [3, 3], rate=6, activation_fn=None) + x

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_6], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)
        atrous_pool_block_12 = slim.conv2d(x, depth/2, [3, 3], rate=12, activation_fn=None) + x

        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_3, atrous_pool_block_6, atrous_pool_block_12), axis=3)

        aspp_feat = self.nn.conv(net, depth, 3, 1)

        return aspp_feat   

    def asppcsr_(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (3, 6, 12) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """
        num_bolck_=1
        bn=slim.batch_norm 
        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth/2, [1, 1], activation_fn=None)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        inputs = self.nn.maxpool(inputs, 3)  

        inputs_feat = tf.concat([image_features, inputs], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)
        atrous_pool_block_1 = self.resAtrousConv(x, depth/2, kernel_size=1, rate=1, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_1], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)        
        atrous_pool_block_3 = self.resAtrousConv(x, depth/2, kernel_size=3, rate=3, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_3], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)        
        atrous_pool_block_6 = self.resAtrousConv(x, depth/2, kernel_size=3, rate=6, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_6], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)
        atrous_pool_block_12 = self.resAtrousConv(x, depth/2, kernel_size=3, rate=12, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_3, atrous_pool_block_6, atrous_pool_block_12), axis=3)

        aspp_feat = self.nn.conv(net, depth, 3, 1)

        return aspp_feat           

    def asppcsr2(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (3, 6, 12) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """
        num_bolck_=2
        bn=slim.batch_norm 
        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth/2, [1, 1], activation_fn=None)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        inputs_feat = tf.concat([image_features, inputs], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)
        atrous_pool_block_1 = self.resAtrousConv(x, depth/2, kernel_size=1, rate=1, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_1], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)        
        atrous_pool_block_3 = self.resAtrousConv(x, depth/2, kernel_size=3, rate=3, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_3], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)        
        atrous_pool_block_6 = self.resAtrousConv(x, depth/2, kernel_size=3, rate=6, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_6], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)
        atrous_pool_block_12 = self.resAtrousConv(x, depth/2, kernel_size=3, rate=12, num_bolck=num_bolck_, activation_fn=tf.nn.elu)


        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_3, atrous_pool_block_6, atrous_pool_block_12), axis=3)

        aspp_feat = self.nn.conv(net, depth, 3, 1)

        return aspp_feat  

    def asppcsre2(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (3, 6, 12) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """
        num_bolck_=2
        bn=slim.batch_norm 
        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth/2, [1, 1], activation_fn=None)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        inputs_feat = tf.concat([image_features, inputs], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)
        atrous_pool_block_1 = self.resAtrousConve(x, depth/2, kernel_size=1, rate=1, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_1], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)        
        atrous_pool_block_3 = self.resAtrousConve(x, depth/2, kernel_size=3, rate=3, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_3], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)        
        atrous_pool_block_6 = self.resAtrousConve(x, depth/2, kernel_size=3, rate=6, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_6], 3)
        x = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)
        atrous_pool_block_12 = self.resAtrousConve(x, depth/2, kernel_size=3, rate=12, num_bolck=num_bolck_, activation_fn=tf.nn.elu)

        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_3, atrous_pool_block_6, atrous_pool_block_12), axis=3)

        aspp_feat = self.nn.conv(net, depth, 3, 1)

        return aspp_feat          

    def asppcsre(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (3, 6, 12) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """
        bn=slim.batch_norm 
        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth/2, [1, 1], activation_fn=tf.nn.elu)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))
             
        inputs_feat = tf.concat([image_features, inputs], 3)
        shortcut = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)
        atrous_pool_block_1_0 = slim.conv2d(inputs_feat, depth/2, [1, 1], activation_fn=tf.nn.elu) 
        atrous_pool_block_1 = slim.conv2d(atrous_pool_block_1_0, depth/2, [1, 1], normalizer_fn = bn, activation_fn=tf.nn.elu) + shortcut

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_1], 3)
        shortcut = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)
        atrous_pool_block_3_0 = slim.conv2d(inputs_feat, depth/2, [3, 3], rate=3, activation_fn=tf.nn.elu) 
        atrous_pool_block_3 = slim.conv2d(atrous_pool_block_3_0, depth/2, [3, 3], normalizer_fn = bn, rate=3, activation_fn=tf.nn.elu) + shortcut

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_3], 3)
        shortcut = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)
        atrous_pool_block_6_0 = slim.conv2d(inputs_feat, depth/2, [3, 3], rate=6, activation_fn=tf.nn.elu)         
        atrous_pool_block_6 = slim.conv2d(atrous_pool_block_6_0, depth/2, [3, 3], normalizer_fn = bn, rate=6, activation_fn=tf.nn.elu) + shortcut

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_6], 3)
        shortcut = self.nn.conv(inputs_feat, depth/2, 3, 1, normalizer_fn = bn, activation_fn=tf.nn.elu)
        atrous_pool_block_12_0 = slim.conv2d(inputs_feat, depth/2, [3, 3], rate=12, activation_fn=tf.nn.elu) 
        atrous_pool_block_12 = slim.conv2d(atrous_pool_block_12_0, depth/2, [3, 3], normalizer_fn = bn, rate=12, activation_fn=tf.nn.elu) + shortcut


        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_3, atrous_pool_block_6, atrous_pool_block_12), axis=3)

        aspp_feat = self.nn.conv(net, depth, 3, 1)

        return aspp_feat  

    def asppcso(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (3, 6, 12) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """

        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth/2, [1, 1], activation_fn=None)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        inputs_feat = tf.concat([image_features, inputs], 3)
        atrous_pool_block_1 = slim.conv2d(inputs_feat, depth/2, [1, 1], activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_1], 3)
        atrous_pool_block_6 = slim.conv2d(inputs_feat, depth/2, [3, 3], rate=6, activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_6], 3)
        atrous_pool_block_12 = slim.conv2d(inputs_feat, depth/2, [3, 3], rate=12, activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_12], 3)
        atrous_pool_block_18 = slim.conv2d(inputs_feat, depth/2, [3, 3], rate=18, activation_fn=None)


        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_6, atrous_pool_block_12, atrous_pool_block_18), axis=3)

        aspp_feat = self.nn.conv(net, depth, 3, 1)

        return aspp_feat 

    def asppcsr_old(self, inputs, depth=256):
        """
        ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (3, 6, 12) when output stride = 16
        (all with 256 filters and batch normalization), and (b) the image-level features as described in the paper
        """

        feature_map_size = tf.shape(inputs)

        # Global average pooling
        image_features = tf.reduce_mean(inputs, [1, 2], keep_dims=True)

        image_features = slim.conv2d(image_features, depth/2, [1, 1], activation_fn=None)
        image_features = tf.compat.v1.image.resize_bilinear(image_features, (feature_map_size[1], feature_map_size[2]))

        inputs_feat = tf.concat([image_features, inputs], 3)
        atrous_pool_block_1 = slim.conv2d(inputs_feat, depth/2, [1, 1], activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_1], 3)
        atrous_pool_block_3 = slim.conv2d(inputs_feat, depth/2, [3, 3], rate=3, activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_3], 3)
        atrous_pool_block_6 = slim.conv2d(inputs_feat, depth/2, [3, 3], rate=6, activation_fn=None)

        inputs_feat = tf.concat([inputs_feat, atrous_pool_block_6], 3)
        atrous_pool_block_12 = slim.conv2d(inputs_feat, depth/2, [3, 3], rate=12, activation_fn=None)


        net = tf.concat((image_features, atrous_pool_block_1, atrous_pool_block_3, atrous_pool_block_6, atrous_pool_block_12), axis=3)

        #aspp_feat = self.nn.bottleneckblock(net, depth, 1)
        aspp_feat = self.nn.resconv(net, depth, 1)   # mode - 2      

        return aspp_feat              