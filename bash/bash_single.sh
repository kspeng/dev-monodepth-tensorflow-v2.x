#!/bin/bash
encoder_='vggASPP'
batch_=8
epoch_=20
epoch_=100
dataset_='kitti'
split_='eigen'
ckpt_=72500 # 
ckpt_=362500 #  	 
image_path_='./data/test/00.jpg'
output_path_='./data/test/'

model_name="$(printf '%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
echo ">>> ${model_name}"

python tfcv_main.py --mode single \
--encoder ${encoder_} \
--image_path ${image_path_} \
--output_directory ${output_path_} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%})



