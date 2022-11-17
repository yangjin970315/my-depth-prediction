#!/bin/bash

MODEL_CONFIG="configs/models/StructRankNetPQRS_res101.yaml"
DATASET_CONFIG="configs/datasets/KITTI_eigen.yaml"
#DATASET_CONFIG="configs/datasets/NYU_Depth_v2.yaml"
EXP_CONFIG="configs/default.yaml"
#EXP_CONFIG="experiments/configs/exp1.yaml"

python3 -u trainer.py --test --model_config ${MODEL_CONFIG} --dataset_config ${DATASET_CONFIG} --exp_config ${EXP_CONFIG}
