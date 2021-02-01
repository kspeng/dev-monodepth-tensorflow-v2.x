#!/bin/bash
encoder_='vggASPPcs'
decoder_='bts'
batch_=8
epoch_=100
dataset_='kitti'
ckpt_=110000 # 362500

model_name="$(printf '%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
echo ">>> ${model_name}"

python tfcv_main.py --mode train \
--data_path "$(printf './data/dataset/%s/data/' ${dataset_%})" \
--filenames_file "$(printf './data/filenames/%s_train_files.txt' ${dataset_%})" \
--log_directory models/ \
--model_name ${model_name} \
--dataset ${dataset_} \
--encoder ${encoder_} \
--decoder ${decoder_} \
--batch_size ${batch_} \
--num_epochs ${epoch_} \
--height 256 \
--width 512
#--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%}) #\
#--do_reg \
#--retrain 

