'''
MODEL    :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-24
FILE     :: utils_monodepth.py 
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils.bilinear_sampler import *
from nn.nn_kits import NnKits
from utils.img_proc import ImgProc

class MdeProc(object):
    """utils for monodepth"""

    def __init__(self, img):
        self.nn = NnKits()
        self.imgproc = ImgProc()
        self.img = img

        # init 
        if img is not None:
            self.geo_reg_init()

    # Regularization
    def geo_reg_init(self):
        b = self.img.get_shape().as_list()[0]
        h = self.img.get_shape().as_list()[1]
        w = self.img.get_shape().as_list()[2]
        m, l = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = np.clip(2*(0.5 - l), 0, 1)
        m_mask = 1. - np.clip(2.*abs(0.5 - m), 0, 1)

        geo_reg_mask = l_mask * m_mask         
        geo_reg_mask = np.repeat(geo_reg_mask[np.newaxis, :, :], b, axis=0)    
        geo_reg_mask = np.repeat(geo_reg_mask[:, :, :, np.newaxis], 2, axis=3)               
        self.geo_reg_mask  = self.imgproc.scale_pyramid(geo_reg_mask,  4)     


    # Disparity Generation
    def get_disp(self, x, do_due=False):
        if do_due:
            disp = 0.3 * self.nn.conv(x, 4, 3, 1, tf.nn.sigmoid)
        else:
            disp = 0.3 * self.nn.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp        

    # Image Conversion by Disparity
    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    # Smoothness
    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.imgproc.gradient_x(d) for d in disp]
        disp_gradients_y = [self.imgproc.gradient_y(d) for d in disp]
        image_gradients_x = [self.imgproc.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.imgproc.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def get_disparity_edge(self, disp, pyramid):
        disp_edge   = [tf.image.sobel_edges(d) for d in disp]
        image_edge  = [tf.image.sobel_edges(img) for img in pyramid]

        disp_edge_x   = [tf.abs(d[:,:,:,:,0]) for d in disp_edge]
        image_edge_x  = [tf.abs(img[:,:,:,:,0]) for img in image_edge]
        
        weights_x   = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_edge_x]        
        edgeness_x = [disp_edge_x[i] * weights_x[i] for i in range(4)]

        return edgeness_x 

    # occlusion detection
    def get_disp_occ(self, disp, kernel_size=21):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        
        disp_p = tf.pad(disp, [[0, 0], [1, 1], [p, p], [0, 0]])
        disp_p_ap = tf.nn.avg_pool2d(disp_p, ksize=[1, 3, p, 1], strides=[1, 1, 1, 1], padding='SAME')

        gx = disp_p_ap[:,1:-1,:(-2*p),:] - disp_p_ap[:,1:-1,(p+1):(-p+1),:]

        gx_max, gx_min = tf.reduce_max(gx), tf.reduce_min(gx)

        edgeness = tf.sigmoid((((gx - gx_min) / ( gx_max -  gx_min))-0.5)*32)

        occ = tf.nn.relu(0.5 - edgeness)

        return occ