#!/bin/bash
encoder_='vggASPP'
batch_=8
epoch_=50
dataset_='cityscapes'
#ckpt_=130000


model_name="$(printf '%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
echo ">>> Training stage 1 - ${model_name}"

#singularity run --nv ~/workspace/envImg/tfcvpy36tf15.img \
python tfcv_main.py --mode train \
--data_path "$(printf '../../dataset/%s/train/' ${dataset_%})" \
--filenames_file "$(printf './data/filenames/%s_train_files.txt' ${dataset_%})" \
--log_directory models/ \
--model_name ${model_name} \
--dataset ${dataset_} \
--encoder ${encoder_} \
--batch_size ${batch_} \
--num_epochs ${epoch_} #\
#--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%}) \

dataset_2='kitti'
ckpt_=143600
model_name_2="$(printf '%s_%s_%dx%d' ${dataset_2%} ${encoder_%} ${batch_%} ${epoch_%})"
echo ">>> Training stage 2 - ${model_name_2}"

#singularity run --nv ~/workspace/envImg/tfcvpy36tf15.img \
python tfcv_main.py --mode train \
--data_path "$(printf '../../dataset/%s/data/' ${dataset_2%})" \
--filenames_file "$(printf './data/filenames/%s_train_files.txt' ${dataset_2%})" \
--log_directory models/ \
--model_name ${model_name_2} \
--dataset ${dataset_2} \
--encoder ${encoder_} \
--batch_size ${batch_} \
--num_epochs ${epoch_} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%}) \
--retrain
#--checkpoint_path $(printf 'models/%s/model-%d' ${model_name_2%} ${ckpt_%})

ckpt_=181250
epoch_=20

model_name_3="$(printf '%s_%s_%dx%d_regh' ${dataset_2%} ${encoder_%} ${batch_%} ${epoch_%})"
echo ">>> Training stage 3 - ${model_name_3}"

#singularity run --nv ~/workspace/envImg/tfcvpy36tf15.img \
python tfcv_main.py --mode train \
--data_path "$(printf '../../dataset/%s/data/' ${dataset_2%})" \
--filenames_file "$(printf './data/filenames/%s_train_files.txt' ${dataset_2%})" \
--log_directory models/ \
--model_name ${model_name_3} \
--dataset ${dataset_2} \
--encoder ${encoder_} \
--batch_size ${batch_} \
--num_epochs ${epoch_} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name_2%} ${ckpt_%}) \
--retrain \
--do_reg
#--checkpoint_path $(printf 'models/%s/model-%d' ${model_name_3%} ${ckpt_%})
