#!/bin/bash

# Dataset configuration
SCENE=chair
ALL_TRAIN_TRANSFORM=${BLENDER_DATASET_PATH}/${SCENE}/split/${SPLIT_NUM}/transforms_train.json # SET TO YOUR FOLDER
TEST_TRANSFORM=${BLENDER_DATASET_PATH}/${SCENE}/transforms_test.json # SET TO YOUR FOLDER

# Experiment configuration
REP=1
EXP=VMF
CHECKPOINT_DIR=${RESULT_PATH}/nerf_director/${SCENE}/${EXP}/${REP}

# Running
python nerf_director.py --rep ${REP} \
                        --model ${SCENE} \
                        --sampling vmf \
                        --all_train_transform ${ALL_TRAIN_TRANSFORM} \
                        --test_transform ${TEST_TRANSFORM} \
                        --checkpoint_dir ${CHECKPOINT_DIR} \
                        --vmf_sigma 0.25 \
                        --vmf_kappa 100 \
                        --lloyd \