#!/bin/bash
encoder_='vggASPP'
batch_=8
epoch_=90
dataset_='kitti'
split_='eigen'
ckpt_=72500 # 
ckpt_=326250 #  	 

model_name="$(printf 'c_%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
echo ">>> ${model_name}"

python tfcv_main.py --mode test \
--data_path ./data/dataset/Deepdrive/ \
--filenames_file ./data/filenames/deepdrive_test_files.txt \
--dataset cityscapes \
--encoder ${encoder_} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%})
