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
import scipy.misc
import cv2

from engine.mde.mde_model import *
from data.dataloader import *
from nn.average_gradients import *

class Test:

    def __init__(self, params):
        self.params = params

    def count_text_lines(self, file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        return len(lines)

    def test(self):
        """Test function."""
        params = self.params

        dataloader = MonodepthDataloader(params.data_path, params.filenames_file, params, params.dataset, params.mode)
        left  = dataloader.left_image_batch
        right = dataloader.right_image_batch

        model = MonodepthModel(params, params.mode, left, right)

        # SESSION
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True    
        sess = tf.compat.v1.Session(config=config)

        # SAVER
        train_saver = tf.compat.v1.train.Saver()

        # INIT
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # RESTORE
        if params.checkpoint_path == '':
            restore_path = tf.train.latest_checkpoint(params.log_directory + '/' + params.model_name)
        else:
            restore_path = params.checkpoint_path.split(".")[0]
        train_saver.restore(sess, restore_path)

        num_test_samples = self.count_text_lines(params.filenames_file)

        #print('writing disparities.')
        if params.output_directory == '':
            output_directory = os.path.dirname(params.checkpoint_path)
        else:
            output_directory = params.output_directory

        print('now testing {} files'.format(num_test_samples))
        disparities     = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
        disparities_pp  = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
        disparities_ppp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
        disparities_ampp  = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)    
        comp_tower  = []
        pp_comp_tower  = []
        ppp_comp_tower  = []
        ampp_comp_tower  = []                        
        comp_offset = 10 # avoid unstable loading efficiency of first few test samples

        native_test = not True
        if native_test:
            for step in range(num_test_samples):
                #st = time.time()
                src_ = sess.run(model.left[0])
                #src[step]       = src_
            #np.save(output_directory + '/src.npy', src)
            print('Pilot run done.')        
            for step in range(num_test_samples):
                st = time.time()
                disp = sess.run(model.disp_est)
                if step > comp_offset:
                    comp_tower += time.time() - st,
                #disparities[step]       = disp
            total_time      = sum(comp_tower)
            print('Native Inferece FPS: ', round((num_test_samples-comp_offset)/total_time, 2))
        else:

            if not (params.mode == 'demo'):
                for step in range(num_test_samples):
                    st = time.time()
                    disp = sess.run(model.disp_est)
                    if step > comp_offset:
                        comp_tower += time.time() - st,
                    disparities[step]       = disp
                total_time      = sum(comp_tower)
                print('Native Inferece FPS: ', round((num_test_samples-comp_offset)/total_time, 2))
                np.save(output_directory + '/disparities.npy',    disparities)

                
                for step in range(num_test_samples):
                    st = time.time()
                    disp_pp = sess.run(model.disp_est_pp)
                    if step > comp_offset:
                        pp_comp_tower += time.time() - st,
                    disparities_pp[step]    = disp_pp
                pp_total_time   = sum(pp_comp_tower)
                print('PP Inferece FPS: ', round((num_test_samples-comp_offset)/pp_total_time, 2))
                np.save(output_directory + '/disparities_pp.npy', disparities_pp)


            for step in range(num_test_samples):
                st = time.time()
                disp_ppp = sess.run(model.disp_est_ppp)
                if step > comp_offset:
                    ppp_comp_tower += time.time() - st,
                disparities_ppp[step]   = disp_ppp
            ppp_total_time  = sum(ppp_comp_tower)
            print('EG-PP Inferece FPS: ', round((num_test_samples-comp_offset)/ppp_total_time, 2))
            if not (params.mode == 'demo'):
                np.save(output_directory + '/disparities_ppp.npy', disparities_ppp)
            else:
                np.save(output_directory + '/disparities_demo_ppp.npy', disparities_ppp)

            for step in range(num_test_samples):
                st = time.time()
                disp_ampp = sess.run(model.disp_est_ampp)
                if step > comp_offset:
                    ampp_comp_tower += time.time() - st,
                disparities_ampp[step]    = disp_ampp
            ampp_total_time = sum(ampp_comp_tower)
            print('AM-PP Inferece FPS: ', round((num_test_samples-comp_offset)/ampp_total_time, 2))
            np.save(output_directory + '/disparities_ampp.npy', disparities_ampp)

        print('done.')
        #print('Total time: ', round(total_time, 2))    


        
        print('Save file done.')        