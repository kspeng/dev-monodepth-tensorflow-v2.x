'''
MODEL   :: Tensorflow Computer Vision Platform
DATE    :: 2020-05-31
FILE    :: mde loss function
'''

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from nn.nn_kits import NnKits
from engine.mde.img_proc import ImgProc
from engine.mde.mde_proc import MdeProc


class MdeLoss(object):
    def __init__(self):
        pass

    def init_model_output(self, mde_model):
        self.mde_model = mde_model

    def get_image_loss(self):
        # IMAGE RECONSTRUCTION
        # L1
        self.l1_left    = [tf.abs( self.mde_model.left_est[i] - self.mde_model.left_pyramid[i]) for i in range(4)]
        self.l1_reconstruction_loss_left  = [tf.reduce_mean(l) for l in self.l1_left]
        self.l1_right   = [tf.abs(self.mde_model.right_est[i] - self.mde_model.right_pyramid[i]) for i in range(4)]
        self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

        # SSIM
        self.ssim_left  = [self.mde_model.imgproc.SSIM( self.mde_model.left_est[i],  self.mde_model.left_pyramid[i]) for i in range(4)]
        self.ssim_loss_left  = [tf.reduce_mean(s) for s in self.ssim_left]
        self.ssim_right = [self.mde_model.imgproc.SSIM(self.mde_model.right_est[i], self.mde_model.right_pyramid[i]) for i in range(4)]
        self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

        # WEIGTHED SUM
        self.image_loss_right = [self.mde_model.params.alpha_image_loss * self.ssim_loss_right[i] + \
                                (1 - self.mde_model.params.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
        self.image_loss_left  = [self.mde_model.params.alpha_image_loss * self.ssim_loss_left[i]  + \
                                (1 - self.mde_model.params.alpha_image_loss) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
        self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

        return self.image_loss

    def get_lr_losss(self):
        # LR CONSISTENCY
        self.lr_left_loss   = [tf.reduce_mean(tf.abs(self.mde_model.right_to_left_disp[i] - self.mde_model.disp_left_est[i]))  for i in range(4)]
        self.lr_right_loss  = [tf.reduce_mean(tf.abs(self.mde_model.left_to_right_disp[i] - self.mde_model.disp_right_est[i])) for i in range(4)]
        self.lr_loss    = tf.add_n(self.lr_left_loss + self.lr_right_loss)
        return self.lr_loss

    def get_dist_loss(self):
        # DISPARITY EDGE
        self.disp_left_loss     = [tf.reduce_mean(tf.abs(self.mde_model.disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
        self.disp_right_loss    = [tf.reduce_mean(tf.abs(self.mde_model.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
        self.disp_loss  = tf.add_n(self.disp_left_loss + self.disp_right_loss)
        return self.disp_loss

    def get_occ_loss(self):
        filter_size = [21, 17, 15, 13]
        self.disp_left_occ  = [tf.reduce_mean((1 + 0.1*self.mde_model.mdeproc.get_disp_occ(self.mde_model.disp_left_est[i], filter_size[i]))*self.mde_model.disp_left_est[i]) for i in range(4)]
        self.disp_right_occ = [tf.reduce_mean((1 + 0.1*self.mde_model.mdeproc.get_disp_occ(self.mde_model.disp_right_est[i], filter_size[i]))*self.mde_model.disp_right_est[i]) for i in range(4)]
        self.occ_loss   = tf.add_n(self.mde_model.disp_left_occ + self.mde_model.disp_right_occ)
        return self.occ_loss

    def get_geo_reg_loss(self):
        self.geo_reg_left   = [tf.reduce_mean(self.mde_model.mdeproc.geo_reg_mask[i] * (1-self.mde_model.disp_left_est[i]) * self.mde_model.disp_left_est[i]) for i in range(4)]
        self.geo_reg_right  = [tf.reduce_mean(self.mde_model.mdeproc.geo_reg_mask[i] * (1-self.mde_model.disp_right_est[i]) * self.mde_model.disp_right_est[i]) for i in range(4)]                      
        self.geo_reg_loss   = tf.add_n(self.post_left_reg + self.post_right_reg)
        return self.geo_reg_loss
      
