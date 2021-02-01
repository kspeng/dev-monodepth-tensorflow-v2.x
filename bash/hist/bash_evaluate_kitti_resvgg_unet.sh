#!/bin/bash
encoder_='resvggASPP'
decoder_='unet'
decoder_up_type_='vgg'
batch_=8
epoch_=100
#epoch_=20
dataset_='kitti'
split_='kitti'
ckpt_=362500 # 
#ckpt_=72500 #

model_name="$(printf '%s_%s_%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${decoder_%} ${decoder_up_type_%} ${batch_%} ${epoch_%})"
echo ">>> ${model_name}"

python tfcv_main.py --mode test \
--data_path ./data/dataset/kitti/stereo2015/ \
--filenames_file ./data/filenames/kitti_stereo_2015_test_files.txt \
--log_directory ../log/ \
--encoder ${encoder_} \
--encoder ${encoder_} \
--decoder_up_type ${decoder_up_type_} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%})

echo ">>> ${backbone%}"
echo ">>> Kitti: Native Evaluation"
python ./utils/eval/evaluate_kitti.py --split ${split_} \
--gt_path ./data/dataset/kitti/stereo2015/ \
--predicted_disp_path $(printf 'models/%s/disparities.npy' ${model_name%})

echo ">>> Kitti: Post-Processing Evaluation"
python ./utils/eval/evaluate_kitti.py --split ${split_} \
--gt_path ./data/dataset/kitti/stereo2015/ \
--predicted_disp_path $(printf 'models/%s/disparities_pp.npy' ${model_name%})

echo ">>> Kitti: Edge-Guided Post-Processing Evaluation"
python ./utils/eval/evaluate_kitti.py --split ${split_} \
--gt_path ./data/dataset/kitti/stereo2015/ \
--predicted_disp_path $(printf 'models/%s/disparities_ppp.npy' ${model_name%})

echo ">>> Kitti: Adv Minimum Post-Processing Evaluation"
python ./utils/eval/evaluate_kitti.py --split ${split_} \
--gt_path ./data/dataset/kitti/stereo2015/ \
--predicted_disp_path $(printf 'models/%s/disparities_ampp.npy' ${model_name%})
