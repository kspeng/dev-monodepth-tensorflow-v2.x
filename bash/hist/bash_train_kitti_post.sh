#!/bin/bash
encoder_='vggASPPc'
batch_=8
epoch_=20
dataset_='kitti'
ckpt_=362500

model_name="$(printf '%s_%s_%dx%d_regh' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
echo ">>> ${model_name}"

python tfcv_main.py --mode train \
--data_path "$(printf './data/dataset/%s/data/' ${dataset_%})" \
--filenames_file "$(printf './data/filenames/%s_train_files.txt' ${dataset_%})" \
--log_directory models/ \
--model_name ${model_name} \
--dataset ${dataset_} \
--encoder ${encoder_} \
--batch_size ${batch_} \
--num_epochs ${epoch_} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%}) \
--do_reg \
--retrain 

