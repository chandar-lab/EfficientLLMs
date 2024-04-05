#!/bin/bash
#SBATCH --job-name=gpt2_1gpu
#SBATCH --time=0-03:00

#SBATCH --partition=unkillable    # ask for unkillable job
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6              # number of cpus per gpu
#SBATCH --mem-per-gpu=32G               # memory per gpu
#SBATCH --signal=TERM@60                # SIGTERM 60s prior to the allocation's end

export SLURM_CPUS_PER_GPU=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU

export CUBLAS_WORKSPACE_CONFIG=:16:8
export HF_HOME=$SCRATCH/hf_home
module load libffi
source ../ENV/bin/activate


python src/run_eval.py --configs 'configs/base.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet' \
--tasks 'commitmentbank,copa,wic,boolq' --num_fewshot 2 --provide_description --seed 1234
