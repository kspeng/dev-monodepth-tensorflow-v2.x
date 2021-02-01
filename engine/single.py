'''
MODEL    :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-23
FILE     :: test.py 
'''

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import cv2
import imageio
from PIL import Image
import PIL.Image as pil
import matplotlib.pyplot as plt
from engine.mde_model import *
from data.dataloader import *
from nn.average_gradients import *

class Single:

    def __init__(self, params):
        self.params = params

    def count_text_lines(self, file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        return len(lines)

    def single(self):
        """Test function."""
        params = self.params

        left  = tf.placeholder(tf.float32, [2, params.height, params.width, 3])        
        model = MonodepthModel(params, params.mode, left, None)

        input_image = imageio.imread(params.image_path)
        #input_image = scipy.misc.imread(params.image_path, mode="RGB")
        original_height, original_width, num_channels = input_image.shape
        #input_image = scipy.misc.imresize(input_image, [params.input_height, params.input_width], interp='lanczos')
        input_image = np.array(Image.fromarray(input_image).resize([params.width, params.height], Image.BILINEAR))#LANCZOS))  
        
        input_image = input_image.astype(np.float32) / 255
        input_images = np.stack((np.fliplr(input_image), input_image), 0)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True    
        sess = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # RESTORE
        if params.checkpoint_path == '':
            restore_path = tf.train.latest_checkpoint(params.log_directory + '/' + params.model_name)
        else:
            restore_path = params.checkpoint_path.split(".")[0]
        train_saver.restore(sess, restore_path)

        #num_test_samples = self.count_text_lines(params.filenames_file)

        #print('writing disparities.')
        if params.output_directory == '':
            output_directory = os.path.dirname(params.checkpoint_path)
        else:
            output_directory = params.output_directory
        '''
        disp = sess.run(model.disp_est, feed_dict={left: input_images})
        disp_to_img = np.array(Image.fromarray(disp.squeeze()).resize([original_width,original_height], Image.BILINEAR))#LANCZOS)) 
        fiename = params.image_path[:-4]
        output_path = "{}_disp.png".format(fiename)
        plt.imsave(output_path, disp_to_img, cmap='plasma')
        '''
        #disp = sess.run(model.occ_map_gx_r, feed_dict={left: input_images})
        #disp = sess.run(model.disp_est_right, feed_dict={left: input_images})        
        disp = sess.run(model.disp_est_right, feed_dict={left: input_images})        
        disp_to_img = np.array(Image.fromarray(disp.squeeze()).resize([original_width,original_height], Image.BILINEAR))#LANCZOS)) 
        
        
        #disp_to_img = 0.5 - disp_to_img
        #disp_to_img = np.clip(disp_to_img, 0, 1)
        fiename = params.image_path[:-4]
        #output_path = "{}_right.png".format(fiename)
        #output_path = "{}_occ_map_gx_l.png".format(fiename)
        #output_path = "{}_occ_map_gx_r.png".format(fiename)
        disp_to_img = np.fliplr(disp_to_img)
        output_path = "{}_left_flip.png".format(fiename)
        plt.imsave(output_path, disp_to_img, cmap='plasma')


        print('Save file done.')        