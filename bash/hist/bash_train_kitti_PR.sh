#!/bin/bash
encoder_='resASPP'
batch_=8
epoch_=100
epoch_pr=20
dataset_='kitti'
ckpt_=362500 #181250 # 362500

model_name_src="$(printf '%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
model_name_pr="$(printf '%s_%s_%dx%d_pr' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_pr%})"
echo ">>> ${model_name_pr}"

python tfcv_main.py --mode train \
--data_path "$(printf './data/dataset/%s/data/' ${dataset_%})" \
--filenames_file "$(printf './data/filenames/%s_train_files.txt' ${dataset_%})" \
--log_directory models/ \
--model_name ${model_name_pr} \
--dataset ${dataset_} \
--encoder ${encoder_} \
--batch_size ${batch_} \
--num_epochs ${epoch_pr} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name_src%} ${ckpt_%}) \
--do_reg \
--retrain 


encoder_='vggASPP'
batch_=8
epoch_=90
epoch_pr=20
dataset_='kitti'
ckpt_=326250 # 362500


model_name_src="$(printf 'c_%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
model_name_pr="$(printf 'c_%s_%s_%dx%d_pr' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_pr%})"
echo ">>> ${model_name_pr}"

python tfcv_main.py --mode train \
--data_path "$(printf './data/dataset/%s/data/' ${dataset_%})" \
--filenames_file "$(printf './data/filenames/%s_train_files.txt' ${dataset_%})" \
--log_directory models/ \
--model_name ${model_name_pr} \
--dataset ${dataset_} \
--encoder ${encoder_} \
--batch_size ${batch_} \
--num_epochs ${epoch_pr} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name_src%} ${ckpt_%}) \
--do_reg \
--retrain 


encoder_='resASPP'
batch_=8
epoch_=90
epoch_pr=20
dataset_='kitti'
ckpt_=326250 # 362500


model_name_src="$(printf 'c_%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
model_name_pr="$(printf 'c_%s_%s_%dx%d_pr' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_pr%})"
echo ">>> ${model_name_pr}"

#singularity run --nv ~/workspace/envImg/tfcvpy36tf15.img \
python tfcv_main.py --mode train \
--data_path "$(printf './data/dataset/%s/data/' ${dataset_%})" \
--filenames_file "$(printf './data/filenames/%s_train_files.txt' ${dataset_%})" \
--log_directory models/ \
--model_name ${model_name_pr} \
--dataset ${dataset_} \
--encoder ${encoder_} \
--batch_size ${batch_} \
--num_epochs ${epoch_pr} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name_src%} ${ckpt_%}) \
--do_reg \
--retrain 