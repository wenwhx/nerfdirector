#!/bin/bash

# Dataset configuration
SCENE=M60
ALL_TRAIN_TRANSFORM=${TNT_DATASET_PATH}/${SCENE}/transform/transforms_train.json # SET TO YOUR FOLDER
TEST_TRANSFORM=${TNT_DATASET_PATH}/${SCENE}/transform/transforms_test.json # SET TO YOUR FOLDER

# Experiment configuration
REP=1
EXP=RANDOM
CHECKPOINT_DIR=${RESULT_PATH}/tnt/${SCENE}/${EXP}/${REP}

# Running
python nerf_director.py --rep ${REP} \
                        --model ${SCENE} \
                        --sampling rs \
                        --real_world_data \
                        --all_train_transform ${ALL_TRAIN_TRANSFORM} \
                        --test_transform ${TEST_TRANSFORM} \
                        --checkpoint_dir ${CHECKPOINT_DIR} \
