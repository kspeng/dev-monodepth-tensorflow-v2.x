#!/bin/bash
encoder_='resnet18ped'
batch_=8
epoch_=100
epoch_=50
dataset_='cityscapes'
split_='kitti'
ckpt_=362500 #181250 # 
ckpt_=72500 #
ckpt_=143600

#model_name="$(printf '%s_%s_%dx%d_%s' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%} ${upconv_%})"
model_name="$(printf '%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
echo ">>> ${model_name}"

python tfcv_main.py --mode test \
--data_path ./data/dataset/make3d/Test134_cropped/ \
--filenames_file ./data/filenames/make3d_test_files.txt \
--log_directory ../log/ \
--encoder ${encoder_} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%})
#--checkpoint_path $(printf 'models/%s/%s' ${model_name%} ${model_name%})

