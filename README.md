# NeRF Director official code release (Instant-NGP codebase)
#### [Project](https://wenwhx.github.io/nerfdirector/) | [Paper](https://arxiv.org/abs/2406.08839)

This repository contains the code for NeRF Director with the Instant-NGP codebase, running on Linux with CUDA 11.7.

## Setup

```
# Create a conda environment
conda create -n nerf_director python=3.8
conda activate nerf_director

# Install pip
conda install pip
pip install --upgrade pip

# Clone the project
git clone git@github.com:wenwhx/nerf_director_test.git

# Install NerfDirector requried packages
cd ../../ # Go back to the working directory of nerf director
pip install -r requirements.txt

# By default, PyTorch3D is not installed with GPU support.
# Please follow the instruction of PyTorch3D.
pip install "git+https://github.com/facebookresearch/pytorch3d.git"


# [Optional] Evaluation of NeRFDirector with instantNGP
# If you would like to evaluate NeRFDirector's selected views please clone our fork of instant ngp
git clone git@github.com:wenwhx/instant-ngp.git

# Follow Instant-NGP's guideline for compilation
cd instant-ngp/

# We tested NeRFDirector with cuda-11.7
export CUDA_PATH=/usr/local/cuda-11.7; cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNGP_BUILD_WITH_GUI=off -DCMAKE_CUDA_COMPILER=${CUDA_PATH}/bin/nvcc
cmake --build build --config RelWithDebInfo
```

Our code depends on **PyTorch3D** (with GPU support), which needs to be installed manually. For their installation issues, we recommend our readers to refer to the official site of [Instant-NGP](https://github.com/NVlabs/instant-ngp) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).


## Dataset Preparation

Our used dataset (both **Blender** and **TanksAndTemples**) can be downloaded [here](https://data.csiro.au/collection/csiro:63796), named as `blender.zip` and `tnt.zip` correspondingly.

### Blender
In our paper, we subsampled each scene's training set into 10 non-overlapping subsets (each contains 300 training views). The splits we used, including `transforms_train.json`, are available in `${BLENDER_DATASET_PATH}/${SCENE}/split/${SPLIT}`. When evaluating each view selection method, we performed 10 runs method, using a different training subset and random seed for training and the same testing set (`${BLENDER_DATASET_PATH}/${SCENE}/transforms_test.json`) for testing. 

### TanksAndTemples
For each scene in `tnt`, we re-splited the training set and testing set. Our used split is in `${TNT_DATASET_PATH}/${SCENE}/transform`. We also provided our used COLMAP reconstruction files: `${TNT_DATASET_PATH}/${SCENE}/colmap_refine/triangulated_model/images.txt`.

## Running
All scripts for our paper experiments are in the folder `./scripts`. One needs to update `ALL_TRAIN_TRANSFORM`, `TEST_TRANSFORM` and `CHECKPOINT_DIR` in each bash script to correctly load the data and save checkpoints. For experiments of FVS based on spatial and photogrammetric distance with the TanksAndTemple dataset, the `COLMAP_LOG` should be specified. For Blender dataset, the `SPLIT_NUM` in each script should also be specified.

### Run on NeRF Blender dataset
```
# fvs
./scripts/blender/run_fvs.sh 

# random
./scripts/blender/run_random.sh 

# vmf
./scripts/blender/run_vmf.sh

# zipf
./scripts/blender/run_zipf.sh 
```

### Run on TanksAndTemple dataset
```
# fvs (only based on spatial distance)
./scripts/tnt/run_fvs.sh 

# fvs (use both spatial and photogrammetric distance)
./scripts/tnt/run_fvs_colmap.sh 

# random
./scripts/tnt/run_random.sh 

# vmf
./scripts/tnt/run_vmf.sh

# zipf
./scripts/tnt/run_zipf.sh 
```

## License and Citation
```
@InProceedings{Xiao:CVPR24:NeRFDirector,
    author    = {Xiao, Wenhui and Santa Cruz, Rodrigo and Ahmedt-Aristizabal, David and Salvado, Olivier and Fookes, Clinton and Lebrat, Leo},
    title     = {NeRF Director: Revisiting View Selection in Neural Volume Rendering},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024}
}
```
Copyright (c) 2024 Commonwealth Scientific and Industrial Research Organisation (CSIRO) ABN 41 687 119 230.

This work is freely available for non-commercial scientific research, non-commercial education, or non-commercial research projects, under the CSIRO Non-Commercial License (based on BSD 3-Clause Clear).

If you wish to use this work for commercial purposes, please contact for approval beforehand: [david.ahmedtaristizabal@data61.csiro.au](mailto:david.ahmedtaristizabal@data61.csiro.au)

