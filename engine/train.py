'''
MODEL   :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-23
FILE    :: train.py 
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
# import tensorflow.contrib.slim as slim
import scipy.misc
import cv2

from engine.mde.mde_model import *
from data.dataloader import *
from nn.average_gradients import *

class Train:

    def __init__(self, params):
        self.params = params

    def count_text_lines(self, file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        return len(lines)

    def train(self):
        """Training loop."""
        params = self.params

        with tf.Graph().as_default(), tf.device('/cpu:0'):

            global_step = tf.Variable(0, trainable=False)

            # OPTIMIZER
            num_training_samples = self.count_text_lines(params.filenames_file)

            steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
            num_total_steps = params.num_epochs * steps_per_epoch
            start_learning_rate = params.learning_rate

            boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
            values = [params.learning_rate, params.learning_rate / 2, params.learning_rate / 4]
            learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries, values)

            opt_step = tf.compat.v1.train.AdamOptimizer(learning_rate)

            print("total number of samples: {}".format(num_training_samples))
            print("total number of steps: {}".format(num_total_steps))

            dataloader = MonodepthDataloader(params.data_path, params.filenames_file, params, params.dataset, params.mode)
            left  = dataloader.left_image_batch
            right = dataloader.right_image_batch

            # split for each gpu
            left_splits  = tf.split(left,  params.num_gpus, 0)
            right_splits = tf.split(right, params.num_gpus, 0)

            tower_grads  = []
            tower_losses = []
            reuse_variables = None
            with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
                for i in range(params.num_gpus):
                    with tf.device('/gpu:%d' % i):

                        model = MonodepthModel(params, params.mode, left_splits[i], right_splits[i], reuse_variables, i)

                        loss = model.total_loss
                        tower_losses.append(loss)

                        reuse_variables = True

                        grads = opt_step.compute_gradients(loss)

                        tower_grads.append(grads)

            grads = average_gradients(tower_grads)

            apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

            total_loss = tf.reduce_mean(tower_losses)

            tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
            tf.summary.scalar('total_loss', total_loss, ['model_0'])
            summary_op = tf.compat.v1.summary.merge_all('model_0')

            # SESSION
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True    
            sess = tf.compat.v1.Session(config=config)

            # SAVER
            summary_writer = tf.compat.v1.summary.FileWriter(params.log_directory + '/' + params.model_name, sess.graph)
            train_saver = tf.compat.v1.train.Saver(max_to_keep=5)

            # COUNT PARAMS
            total_num_parameters = 0
            for variable in tf.compat.v1.trainable_variables():
                total_num_parameters += np.array(variable.get_shape().as_list()).prod()
            print("number of trainable parameters: {}".format(total_num_parameters))

            # INIT
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())
            coordinator = tf.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coordinator)

            # LOAD CHECKPOINT IF SET
            if params.checkpoint_path != '':
                train_saver.restore(sess, params.checkpoint_path.split(".")[0])

                if params.retrain:
                    sess.run(global_step.assign(0))

            # GO!
            start_step = global_step.eval(session=sess)
            start_time = time.time()
            for step in range(start_step, num_total_steps):
                before_op_time = time.time()
                _, loss_value = sess.run([apply_gradient_op, total_loss])
                duration = time.time() - before_op_time
                if step and step % 100 == 0:
                    examples_per_sec = params.batch_size / duration
                    time_sofar = (time.time() - start_time) / 3600
                    training_time_left = (num_total_steps / step - 1.0) * time_sofar
                    print_string = 'batch {:>6}/{} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                    print(print_string.format(step, num_total_steps, examples_per_sec, loss_value, time_sofar, training_time_left))
                    #summary_str = sess.run(summary_op)
                    #summary_writer.add_summary(summary_str, global_step=step)

                path_to_ckpt = './' + params.log_directory + '/' + params.model_name
                if step % 10000 == 0: #10000 == 0:
                    os.makedirs(path_to_ckpt) if not os.path.exists(path_to_ckpt) else None
                    train_saver.save(sess, path_to_ckpt + '/model', global_step=step)

            train_saver.save(sess, path_to_ckpt + '/model', global_step=num_total_steps)

            # FINISHED!
            print('Training Finished.')

