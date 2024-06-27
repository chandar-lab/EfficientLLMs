#!/bin/bash
#SBATCH --job-name=gpt2
#SBATCH --time=0-03:00

#SBATCH --partition=short-unkillable    # ask for unkillable job
#SBATCH --nodes=1                       # number of nodes
#SBATCH --gpus-per-task=4               # number of gpus per node
#SBATCH --cpus-per-task=24              # number of cpus per gpu
#SBATCH --mem-per-gpu=32G               # memory per gpu
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --constraint=80gb               # constraints

export CUBLAS_WORKSPACE_CONFIG=:16:8
export HF_HOME=$SCRATCH/hf_home
module load python/3.8
module load cuda/11.8
module load libffi
source ../ENV/bin/activate

#torchrun --nproc_per_node=4 src/main.py --configs 'configs/base.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet' train
torchrun --nproc_per_node=4 src/main.py --configs 'configs/base.jsonnet, configs/model/llama-small.jsonnet, configs/dataset/openwebtext_llama.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet' train