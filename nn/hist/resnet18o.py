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

class Resnet18(object):
    def __init__(self, is_training):
        self.nn = NnKits()

    def forward(self, x, prefix=''):
        with tf.variable_scope('{}encoder'.format(prefix), reuse=tf.AUTO_REUSE) as scope:
            end_points_collection = scope.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=None,
                                weights_initializer=tf.keras.initializers.he_normal(),
                                biases_initializer=None,
                                activation_fn=None,
                                outputs_collections=end_points_collection):
                print('Building ResNet-18 Model')
                filters = [64, 64, 128, 256, 512]
                kernels = [7, 3, 3, 3, 3]
                strides = [2, 0, 2, 2, 2]

                # conv1
                print('\tBuilding unit: conv1')
                with tf.variable_scope('conv1'):
                    x = self._conv(x, kernels[0], filters[0], strides[0], name='conv1') # [H/2, W/2]
                    x = self._bn(x)
                    x = self._activate(x, type='relu',name='relu1')
                    self.skip1 = x  # [H/2, W/2]
                    x = slim.max_pool2d(x, [3, 3], stride=2, padding='SAME', scope='pool')

                # conv2_x
                x = self._residual_block(x, name='conv2_1')
                x = self._residual_block(x, name='conv2_2')
                self.skip2 = x # [H/4, W/4]

                # conv3_x
                x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
                x = self._residual_block(x, name='conv3_2')
                self.skip3 = x # [H/8, W/8]

                # conv4_x
                x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
                x = self._residual_block(x, name='conv4_2')
                self.skip4 = x # [H/16, W/16]

                # conv5_x
                x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
                x = self._residual_block(x, name='conv5_2')
                # [H/32, W/32]
                self.enc_feat = x
        return self.enc_feat, [self.skip1, self.skip2, self.skip3, self.skip4]

    def _residual_blocks(self, x, out_channel, stride=1, name='unit'):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            print('\tBuilding residual unit: {}'.format(scope.name))
            if in_channel == out_channel:
                if stride == 1:
                    short_cut = tf.identity(x)
                else:
                    short_cut = slim.max_pool2d(x, [stride, stride], stride=stride, padding='SAME', scope='pool')
            else:
                short_cut = self._conv(x, 1, out_channel, stride, name='shortcut')
            # residual
            x = self._conv(x, 3, out_channel, stride, name='conv1')
            x = self._bn(x, name='BatchNorm')
            x = self._activate(x, type='relu', name='relu1')
            x = self._conv(x, 3, out_channel, 1, name='conv2')
            x = self._bn(x, name='BatchNorm_1')

            x = x + short_cut
            x = self._activate(x, type='relu', name='relu2')
            return x

    def _residual_block_first(self, x, out_channel, stride, name='unit'):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            print('\tBuilding residual unit: {}'.format(scope.name))
            if in_channel == out_channel:
                if stride == 1:
                    short_cut = tf.identity(x)
                else:
                    short_cut = slim.max_pool2d(x, [stride, stride], stride=stride, padding='SAME', scope='pool')
            else:
                short_cut = self._conv(x, 1, out_channel, stride, name='shortcut')
            # residual
            x = self._conv(x, 3, out_channel, stride, name='conv1')
            x = self._bn(x, name='BatchNorm')
            x = self._activate(x, type='relu', name='relu1')
            x = self._conv(x, 3, out_channel, 1, name='conv2')
            x = self._bn(x, name='BatchNorm_1')

            x = x + short_cut
            x = self._activate(x, type='relu', name='relu2')
            return x

    def _residual_block(self, x, name='unit'):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            print('\tBuildint residual unit: {}'.format(scope.name))
            short_cut = x
            x= self._conv(x, 3, num_channel, 1, name='conv1')
            x = self._bn(x, name='BatchNorm')
            x = self._activate(x,type='relu', name='relu1')
            x = self._conv(x, 3, num_channel, 1, name='conv2')
            x = self._bn(x, name='BatchNorm_1')

            x = x + short_cut
            x = self._activate(x, type='relu', name='relu2')
            return x

    def _conv_(self, x, filter_size, out_channel, stride, pad='SAME', name='conv'):
        x = slim.conv2d(x, out_channel, [filter_size, filter_size], stride,padding=pad, scope=name)
        return x

    def _conv(self, x, filter_size, out_channel, stride, name='conv', activation_fn=tf.nn.elu):
        pad_size = np.int(filter_size // 2)
        pad_x = tf.pad(x,[[0,0], [pad_size, pad_size], [pad_size, pad_size], [0,0]], mode='REFLECT')
        x = slim.conv2d(pad_x, out_channel, [filter_size, filter_size], stride, padding='VALID', scope=name, activation_fn=activation_fn)
        return x

    def _bn(self, x, name='BatchNorm'):
        #x = slim.batch_norm(x,scale=True,decay=self.decay, epsilon=self.epsilon, is_training=True)#, updates_collections=None)
        #x = tfl.batch_normalization(x,momentum=self.decay,epsilon=self.epsilon,training=self.is_training, name=name, fused=True,reuse=tf.AUTO_REUSE)
        x = slim.batch_norm(x)
        return x

    def _activate(self, x, type='relu', name='relu'):
        if type == 'elu':
            x = tf.nn.elu(x, name=name)
        else:
            x = tf.nn.relu(x, name=name)
        return x
