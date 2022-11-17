#!/bin/bash

MODEL_CONFIG="configs/models/StructRankNetPQRS_res101.yaml"
#DATASET_CONFIG="configs/datasets/NYU_Depth_v2.yaml"
DATASET_CONFIG="configs/datasets/app.yaml"
EXP_CONFIG="configs/default.yaml"

python3 -u app.py --test --model_config ${MODEL_CONFIG} --dataset_config ${DATASET_CONFIG} --exp_config ${EXP_CONFIG}
