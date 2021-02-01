#!/bin/bash
encoder_='vggASPPcs'
batch_=8
epoch_=100
dataset_='cityscapes'
ckpt_=210250

model_name="$(printf '%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
echo ">>> ${model_name}"

python tfcv_main.py --mode train \
--data_path "$(printf './data/dataset/%s/train/' ${dataset_%})" \
--filenames_file "$(printf './data/filenames/%s_train_files.txt' ${dataset_%})" \
--log_directory models/ \
--model_name ${model_name} \
--dataset ${dataset_} \
--encoder ${encoder_} \
--batch_size ${batch_} \
--num_epochs ${epoch_} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%})


