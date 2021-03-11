'''
MODEL	:: Tensorflow Computer Vision Platform
DATE	:: 2020-01-23
FILE 	:: param.py 
'''

from __future__ import absolute_import, division, print_function

import os
import argparse
from config.default import *

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

## Options Class
class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Tensorflow Computer Vision parameters")

        # MODE
        self.parser.add_argument('--mode',                      type=str,   
            help='train or test', default=mode)
        self.parser.add_argument('--model_name',                type=str,   
            help='model name', default=model_name)
        
        # MODEL
        self.parser.add_argument('--do_stereo',                             
            help='if set, will train the stereo model', action=do_stereo)
        self.parser.add_argument('--encoder',                   type=str,   
            help='type of encoder, vgg, resnet50, resvgg, or resaspp', default=encoder)
        self.parser.add_argument('--decoder',                   type=str,   
            help='type of decoder, unet', default=decoder)

        # DATA
        self.parser.add_argument('--dataset',                   type=str,   
            help='dataset to train on, kitti, or cityscapes', default=dataset)
        self.parser.add_argument('--data_path',                 type=str,   
            help='path to the data', default=data_path)        
        self.parser.add_argument('--filenames_file',            type=str,   
            help='path to the filenames text file', default=filenames_file)
        self.parser.add_argument('--image_path',                type=str,   
            help='path to the image', default=image_path)
                                                                            
        # TRAIN                                                                    
        self.parser.add_argument('--height',                    type=int,   
            help='input height', default=height)
        self.parser.add_argument('--width',                     type=int,   
            help='input width', default=width)
                                                                            
        self.parser.add_argument('--batch_size',                type=int,   
            help='batch size', default=batch_size)
        self.parser.add_argument('--num_epochs',                type=int,   
            help='number of epochs', default=num_epochs)
                                                                            
        self.parser.add_argument('--retrain',                               
            help='if used with checkpoint_path, will restart training from step zero', action=retrain)
                                                                    
        # OPTIMIZATION                                                                    
        self.parser.add_argument('--learning_rate',             type=float, 
            help='initial learning rate', default=learning_rate)
                                                                            
        # LOSS                                                                    
        self.parser.add_argument('--lr_loss_weight',            type=float, 
            help='left-right consistency weight', default=lr_loss_weight)
        self.parser.add_argument('--alpha_image_loss',          type=float, 
            help='weight between SSIM and L1 in the image loss', default=alpha_image_loss)
        self.parser.add_argument('--disp_gradient_loss_weight', type=float, 
            help='disparity smoothness weigth', default=disp_gradient_loss_weight)

        # OTHERS
        self.parser.add_argument('--wrap_mode',                 type=str,   
            help='bilinear sampler wrap mode, edge or border', default=wrap_mode)

        # SYSTEM                                                                    
        self.parser.add_argument('--num_gpus',                  type=int,   
            help='number of GPUs to use for training', default=num_gpus)
        self.parser.add_argument('--num_threads',               type=int,   
            help='number of threads to use for data loading', default=num_threads)
        self.parser.add_argument('--full_summary',                          
            help='if set, will keep more data for each summary. Warning: the file can become very large', action=full_summary)
                                                                            
        # PATH                                                                    
        self.parser.add_argument('--output_directory',          type=str,   
            help='output directory for test disparities, if empty outputs to checkpoint folder', default=output_directory)
        self.parser.add_argument('--log_directory',             type=str,   
            help='directory to save checkpoints and summaries', default=log_directory)
        self.parser.add_argument('--checkpoint_path',           type=str,   
            help='path to a specific checkpoint to load', default=checkpoint_path)

    def parse(self):
        self.params = self.parser.parse_args()
        return self.params        
