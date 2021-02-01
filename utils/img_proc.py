'''
MODEL    :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-24
FILE     :: utils_img.py 
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class ImgProc(object):
    """utils for image"""

    def __init__(self):
        pass

    # Post Regularization

    # Image Size
    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.compat.v1.image.resize_area(img, [nh, nw]))
        return scaled_imgs    

    # Image Edge
    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def get_strong_edge_x(self, disp, kernel_size=21):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        disp_p = tf.pad(disp, [[0, 0], [1,1], [p, p], [0, 0]])
        disp_p_ap = tf.nn.avg_pool(disp_p, ksize=[1, 3, p, 1], strides=[1, 1, 1, 1], padding='SAME')
        gx = ( disp_p_ap[:,1:-1:,:(-2*p),:] - disp_p_ap[:,1:-1,(p+1):(-p+1),:] ) / 2
        return gx

    def get_strong_edge_y(self, disp, kernel_size=11):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        disp_p = tf.pad(disp, [[0, 0], [p,p], [1,1], [0, 0]])
        disp_p_ap = tf.nn.avg_pool(disp_p, ksize=[1, p, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        gy = ( disp_p_ap[:,:(-2*p),1:-1,:] - disp_p_ap[:,(p+1):(-p+1),1:-1,:] ) / 2
        return gy    

    # Image Quality
    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)    