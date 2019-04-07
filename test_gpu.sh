#! /bin/bash
source /home/tungnguyen/miniconda3/etc/profile.d/conda.sh
CONDA_CHANGEPS1=false conda activate base

conda activate /home/tungnguyen/miniconda3/envs/hde_gpu
#conda activate hde_gpu

#CUDA_VISIBLE_DEVICES=1 python gpu.py

python test_gpu.py