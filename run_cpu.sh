#! /bin/bash
source /home/tungnguyen/miniconda3/etc/profile.d/conda.sh
CONDA_CHANGEPS1=false conda activate base

conda activate /home/tungnguyen/miniconda3/envs/hde_cpu

#CUDA_VISIBLE_DEVICES=1 python gpu.py

python train_duke_semi.py -c $1
