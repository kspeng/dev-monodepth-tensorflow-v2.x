#!/bin/bash
model_name='adareg' #"$(printf '%s_%s_%dx%d' ${dataset_%} ${encoder_%} ${batch_%} ${epoch_%})"
split_='eigen'
echo ">>> ${model_name}"

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
