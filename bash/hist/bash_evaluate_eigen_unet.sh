#!/bin/bash
encoder_='resvggASPP'
decoder_='unet'
decoder_up_type_='vggitr'
batch_=8
#epoch_=20
epoch_=100
dataset_='kitti'
split_='eigen'
#ckpt_=72500 # 
ckpt_=362500 #  	 

model_name="$(printf '%s_%s_%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${decoder_%} ${decoder_up_type_%} ${batch_%} ${epoch_%})"
echo ">>> ${model_name}"

python tfcv_main.py --mode test \
--data_path ./data/dataset/kitti/data/ \
--filenames_file ./data/filenames/eigen_test_files.txt \
--dataset ${dataset_} \
--encoder ${encoder_} \
--decoder ${decoder_} \
--decoder_up_type ${decoder_up_type_} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%})

echo ">>> ${backbone%}"
echo ">>> Eigen 80: native Evaluation"
python ./utils/eval/evaluate_kitti.py --split ${split_} \
--gt_path ./data/dataset/kitti/data/ \
--max_depth  80 \
--garg_crop \
--predicted_disp_path $(printf './models/%s/disparities.npy' ${model_name%})

echo ">>> Eigen 80: Conventional Post-Processing Evaluation"
python ./utils/eval/evaluate_kitti.py --split ${split_} \
--gt_path ./data/dataset/kitti/data/ \
--max_depth  80 \
--garg_crop \
--predicted_disp_path $(printf './models/%s/disparities_pp.npy' ${model_name%})

echo ">>> Eigen 80: Edge-Guided Post-Processing Evaluation"
python ./utils/eval/evaluate_kitti.py --split ${split_} \
--gt_path ./data/dataset/kitti/data/ \
--max_depth  80 \
--garg_crop \
--predicted_disp_path $(printf 'models/%s/disparities_ppp.npy' ${model_name%})

echo ">>> Eigen 80: Adv Minimum Post-Processing Evaluation"
python ./utils/eval/evaluate_kitti.py --split ${split_} \
--gt_path ./data/dataset/kitti/data/ \
--max_depth  80 \
--garg_crop \
--predicted_disp_path $(printf 'models/%s/disparities_ampp.npy' ${model_name%})

echo ">>> Eigen 50: native Evaluation"
python ./utils/eval/evaluate_kitti.py --split ${split_} \
--gt_path ./data/dataset/kitti/data/ \
--max_depth 50 \
--garg_crop \
--predicted_disp_path $(printf 'models/%s/disparities.npy' ${model_name%})

echo ">>> Eigen 50: Conventional Post-Processing Evaluation"
python ./utils/eval/evaluate_kitti.py --split ${split_} \
--gt_path ./data/dataset/kitti/data/ \
--max_depth 50 \
--garg_crop \
--predicted_disp_path $(printf 'models/%s/disparities_pp.npy' ${model_name%})

echo ">>> Eigen 50: Edge-Guided Post-Processing Evaluation"
python ./utils/eval/evaluate_kitti.py --split ${split_} \
--gt_path ./data/dataset/kitti/data/ \
--max_depth 50 \
--garg_crop \
--predicted_disp_path $(printf 'models/%s/disparities_ppp.npy' ${model_name%})

echo ">>> Eigen 50: Adv Minimum Post-Processing Evaluation"
python ./utils/eval/evaluate_kitti.py --split ${split_} \
--gt_path ./data/dataset/kitti/data/ \
--max_depth 50 \
--garg_crop \
--predicted_disp_path $(printf 'models/%s/disparities_ampp.npy' ${model_name%})