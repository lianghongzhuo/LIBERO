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
robomimic \
einops \
future \
matplotlib \
cloudpickle \
ipython \
rich \
-y

# pip install thop robosuite bddl gym
