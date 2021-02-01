#!/bin/bash
encoder_='vgg'
batch_=8
epoch_=90
dataset_='kitti'
split_='eigen'
ckpt_=72500 # 
ckpt_=326250 #  	 

model_name="$(printf 'c_%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
model_name="model_kitti"
echo ">>> ${model_name}"

python tfcv_main.py --mode demo \
--data_path ./data/dataset/ \
--filenames_file ./data/filenames/kitti_2011_10_03_drive_0034_filenames.txt \
--dataset ${dataset_} \
--encoder ${encoder_} \
--checkpoint_path $(printf 'models/%s/model_kitti' ${model_name%})
#--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%})
