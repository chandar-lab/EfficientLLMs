#!/bin/bash
#SBATCH --job-name=llama-w8a8g8
#SBATCH --time=3-00:00

#SBATCH --partition=lab-chandar
#SBATCH --nodes=1                       # number of nodes
#SBATCH --gpus-per-task=4               # number of gpus per node
#SBATCH --cpus-per-task=24              # number of cpus per gpu
#SBATCH --mem-per-gpu=32G               # memory per gpu
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --signal=TERM@60                # SIGTERM 60s prior to the allocation's end

export CUBLAS_WORKSPACE_CONFIG=:16:8
export HF_HOME=$SCRATCH/hf_home
module load python/3.8
module load cuda/11.8
module load libffi
source ../ENV/bin/activate

torchrun --nproc_per_node=4 src/main.py --configs 'configs/llama_w8a8g8_per_token.jsonnet' train