#!/bin/bash
set -e
eval "$("$MAMBA_EXE" shell hook -s posix)"
if [[ -d $MAMBA_ROOT_PREFIX/envs/py12 ]]; then
    micromamba env remove -n py12 -y
fi
micromamba create -n py12 python=3.12 -y
micromamba activate py12
micromamba install -c conda-forge \
hydra-core \
numpy \
wandb \
easydict \
transformers \
opencv \
einops \
future \
matplotlib \
cloudpickle \
ipython \
rich \
scipy \
pynput \
numba \
pytorch \
torchvision \
h5py \
imageio \
imageio-ffmpeg \
mujoco \
ffmpeg \
qpsolvers \
termcolor \
tqdm \
pytest \
pillow \
gymnasium-all \
hidapi \
huggingface_hub \
transformers \
diffusers \
tensorboard \
tensorboardx \
werkzeug \
-y

pip install thop bddl

cd ../robosuite/ && pip install -e . && \
cd ../robosuite_models/ && pip install -e . && \
cd ../robomimic/ && pip install -e . && \
cd ../LIBERO/ && pip install -e .
