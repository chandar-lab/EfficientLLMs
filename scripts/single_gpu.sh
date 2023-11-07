#!/bin/bash
#SBATCH --job-name=gpt2_1gpu
#SBATCH --time=0-03:00

#SBATCH --partition=short-unkillable    # ask for unkillable job
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4              # number of cpus per gpu
#SBATCH --mem-per-gpu=32G               # memory per gpu

export CUBLAS_WORKSPACE_CONFIG=:16:8
module load libffi
source ../ENV/bin/activate
python src/main.py --configs 'configs/base.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_1gpu.jsonnet' train