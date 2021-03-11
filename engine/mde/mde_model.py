'''
MODEL   :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-24
FILE    :: model.py 
'''

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

# import mde engine 
from engine.mde.img_proc import ImgProc
from engine.mde.mde_proc import MdeProc
from engine.mde.mde_postproc import MdePostproc
from engine.mde.mde_loss import MdeLoss

# import utils 
from utils.bilinear_sampler import *

# import nn kits
from nn.nn_kits import NnKits
from nn.unet import UNet
from nn.vgg import Vgg
from nn.resnet import Resnet
from nn.resnet18 import Resnet18
from nn.resvgg import Resvgg
from nn.resASPP import ResASPP


class MonodepthModel(object):
    """monodepth model"""

    def __init__(self, params, mode, left, right, reuse_variables=None, model_index=0):          
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right

        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables

        # init toolkits 
        self.nn = NnKits()
        self.imgproc = ImgProc()
        self.mdeproc = MdeProc(self.left)
        self.mde_loss = MdeLoss()

        # init codec
        self.encoder(self.params.encoder)
        self.decoder()

        # init postproc
        if not self.mode == 'train':
            self.postproc = MdePostproc(self.left)

        # run model/output
        self.build_model()
        self.build_outputs()

        if not self.mode == 'train':
            return

        # run loss
        self.build_losses()

    # Model Encoder and Decoder
    def encoder(self, encoder_type='vgg'):
        self.stages = 4 
        if encoder_type == 'vgg':
            self.enc = Vgg()
            self.stages = 6        
        elif encoder_type == 'resnet':
            self.enc = Resnet()
            self.stages = 5  
        elif encoder_type == 'resvgg':
            self.enc = Resvgg()
            self.stages = 5
        elif encoder_type == 'resASPP':
            self.enc = ResASPP()  
        elif encoder_type == 'resnet18':
            self.enc = Resnet18()                                                                                      
        else:
            print('>>>{} encoder is not supported. <<<'.format(encoder_type))
            exit()

    def decoder(self): 
        self.dec = UNet(self.stages) 

    def build_model(self):
        with tf.compat.v1.variable_scope('build_model'):
            with tf.compat.v1.variable_scope('model', reuse=self.reuse_variables):

                self.left_pyramid  = self.imgproc.scale_pyramid(self.left,  4)               
                if self.mode == 'train':
                    self.right_pyramid  = self.imgproc.scale_pyramid(self.right, 4)

                if self.params.do_stereo:
                    self.model_input = tf.concat([self.left, self.right], 3)
                else:
                    self.model_input = self.left

                #build model
                self.enc.forward(self.model_input)
                self.dec.forward(self.enc)        

    def build_outputs(self):
        self.disp1, self.disp2, self.disp3, self.disp4 =    self.dec.disp1, \
                                                            self.dec.disp2, \
                                                            self.dec.disp3, \
                                                            self.dec.disp4
        # STORE DISPARITIES
        with tf.compat.v1.variable_scope('disparities'):
            self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]

        if not self.mode == 'train':
            self.disp_est = self.disp1[0,:,:,0]
            self.disp_est_right = self.disp1[0,:,:,1]            
            self.disp_est_pp = self.postproc.post_process(self.disp1)
            self.disp_est_ppp = self.postproc.post_process_plus(self.disp1)
            return

        # GENERATE IMAGES
        with tf.compat.v1.variable_scope('images'):
            self.left_est  = [self.mdeproc.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
            self.right_est = [self.mdeproc.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # LR CONSISTENCY
        with tf.compat.v1.variable_scope('left-right'):
            self.right_to_left_disp = [self.mdeproc.generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(4)]
            self.left_to_right_disp = [self.mdeproc.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

        # DISPARITY SMOOTHNESS
        with tf.compat.v1.variable_scope('smoothness'):
            self.disp_left_smoothness  = self.mdeproc.get_disparity_smoothness(self.disp_left_est,  self.left_pyramid)
            self.disp_right_smoothness = self.mdeproc.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)


    def build_losses(self):
        with tf.compat.v1.variable_scope('losses', reuse=self.reuse_variables):
            self.mde_loss.init_model_output(self)

            self.image_loss = self.mde_loss.get_image_loss()
            self.lr_loss    = self.mde_loss.get_lr_losss()
            self.disp_loss  = self.mde_loss.get_dist_loss()

            # TOTAL LOSS
            self.total_loss = self.image_loss + \
                              self.params.lr_loss_weight * self.lr_loss + \
                              self.params.disp_gradient_loss_weight * self.disp_loss   

            return self.total_loss


