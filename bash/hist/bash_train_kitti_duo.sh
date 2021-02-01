#!/bin/bash
encoder_='vggASPPcs'
batch_=8
epoch_=100
height_=192
width_=640
dataset_='kitti'
ckpt_=68875 # 362500

model_name="$(printf '%s_%s_%dx%d_%dx%d' ${dataset_%} ${encoder_%} ${height_%} ${width_%} ${batch_%} ${epoch_%})"

python tfcv_main.py --mode train \
--data_path "$(printf './data/dataset/%s/data/' ${dataset_%})" \
--filenames_file "$(printf './data/filenames/%s_train_files.txt' ${dataset_%})" \
--log_directory models/ \
--model_name ${model_name} \
--dataset ${dataset_} \
--encoder ${encoder_} \
--batch_size ${batch_} \
--num_epochs ${epoch_} \
--height ${height_} \
--width ${width_} \
--do_duo \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%}) #\
#--do_reg \
#--retrain 

