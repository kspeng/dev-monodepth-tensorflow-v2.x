'''
MODEL   :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-27
FILE    :: unet.py 
'''

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from nn.nn_kits import NnKits
from engine.mde.mde_proc import MdeProc

class UNet(object):

    def __init__(self, stages=5, up_type='conv', do_duo=False, upconv='conv'):
        assert stages >= 4 or stages <= 6, '>>> UNet decoder stages must be within 4 or 6!!'
        self.up_type = up_type
        self.upconv = upconv
        self.do_duo = do_duo
        self.stages = stages
        self.nn = NnKits()
        self.mdeproc = MdeProc(None)
        
    def forward(self, enc):
        self.enc = enc
        # set convenience functions        
        conv   = self.nn.conv
        upconv = self.nn.upconv        
        get_disp = self.mdeproc.get_disp
        upsample_nn = self.nn.upsample_nn

        # set connections
        enc     = self.enc.enc_feat 
        if self.stages == 6:
            skip6   = self.enc.skip6
            skip5   = self.enc.skip5
        if self.stages == 5:
            skip5   = self.enc.skip5
        skip4 = self.enc.skip4
        skip3 = self.enc.skip3
        skip2 = self.enc.skip2
        skip1 = self.enc.skip1

        with tf.compat.v1.variable_scope('decoder'):        
            if self.stages == 6:          
                upconv7 = upconv(enc,   512, 3, 2, type=self.upconv) #H/32
                concat7 = tf.concat([upconv7, skip6], 3)
                iconv7  = conv(concat7,   512, 3, 1)

                upconv6 = upconv(iconv7,   512, 3, 2) #H/32
                concat6 = tf.concat([upconv6, skip5], 3)
                iconv6  = conv(concat6,   512, 3, 1)
            elif self.stages == 5:            
                upconv6 = upconv(enc,   512, 3, 2, type=self.upconv) #H/32
                concat6 = tf.concat([upconv6, skip5], 3)
                iconv6  = conv(concat6,   512, 3, 1)
            else:
                iconv6 = enc

            upconv5 = upconv(iconv6, 256, 3, 2, type=self.upconv) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)

            iconv5  = conv(concat5,  256, 3, 1)
            
            upconv4 = upconv(iconv5, 128, 3, 2, type=self.upconv) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)

            iconv4  = conv(concat4,  128, 3, 1)
            self.disp4 = get_disp(iconv4)
            udisp4  = upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2, type=self.upconv) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,   64, 3, 1)
            self.disp3 = get_disp(iconv3)
            udisp3  = upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,  32, 3, 2, type=self.upconv) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)

            iconv2  = conv(concat2,   32, 3, 1)
            self.disp2 = get_disp(iconv2)
            udisp2  = upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2, type=self.upconv) #H
            concat1 = tf.concat([upconv1, udisp2], 3)

            iconv1  = conv(concat1,   16, 3, 1)
            self.disp1 = get_disp(iconv1)