#!/bin/bash
encoder_='vggASPP'
batch_=1
epoch_=10
dataset_='kitti'
split_='eigen'
ckpt_=326250 # 72500 #   

model_name="$(printf '%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
echo ">>> ${model_name}"


python tfcv_main.py --mode test \
--data_path ../../dataset/kitti/data/ \
--filenames_file dataset/filenames/eigen_test_files.txt \
--dataset ${dataset_} \
--encoder ${encoder_} \
--checkpoint_path $(printf 'models/%s/model-%d' ${model_name%} ${ckpt_%})

echo ">>> ${backbone%}"
echo ">>> Eigen 80: native Evaluation"
python util/evaluate_kitti.py --split ${split_} \
--gt_path ./dataset/kitti/data/ \
--max_depth  80 \
--garg_crop \
--predicted_disp_path $(printf './models/%s/disparities.npy' ${model_name%})

echo ">>> Eigen 80: Conventional Post-Processing Evaluation"
python util/evaluate_kitti.py --split ${split_} \
--gt_path ./dataset/kitti/data/ \
--max_depth  80 \
--garg_crop \
--predicted_disp_path $(printf './models/%s/disparities_pp.npy' ${model_name%})

echo ">>> Eigen 80: Edge-Guided Post-Processing Evaluation"
python util/evaluate_kitti.py --split ${split_} \
--gt_path ./dataset/kitti/data/ \
--max_depth  80 \
--garg_crop \
--predicted_disp_path $(printf 'models/%s/disparities_ppp.npy' ${model_name%})

echo ">>> Eigen 50: native Evaluation"
python util/evaluate_kitti.py --split ${split_} \
--gt_path ./dataset/kitti/data/ \
--max_depth 50 \
--garg_crop \
--predicted_disp_path $(printf 'models/%s/disparities.npy' ${model_name%})

echo ">>> Eigen 50: Conventional Post-Processing Evaluation"
python util/evaluate_kitti.py --split ${split_} \
--gt_path ./dataset/kitti/data/ \
--max_depth 50 \
--garg_crop \
--predicted_disp_path $(printf 'models/%s/disparities_pp.npy' ${model_name%})

echo ">>> Eigen 50: Edge-Guided Post-Processing Evaluation"
python util/evaluate_kitti.py --split ${split_} \
--gt_path ./dataset/kitti/data/ \
--max_depth 50 \
--garg_crop \
--predicted_disp_path $(printf 'models/%s/disparities_ppp.npy' ${model_name%})
