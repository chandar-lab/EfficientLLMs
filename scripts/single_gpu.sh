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
#python src/main.py --configs 'configs/base.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_1gpu.jsonnet' train
srun \
  --nodes=$SLURM_JOB_NUM_NODES \
  --ntasks=$SLURM_JOB_NUM_NODES \
  --cpus-per-gpu=$SLURM_CPUS_PER_GPU \
  --gpus-per-task=$SLURM_GPUS_PER_TASK \
  --ntasks-per-node=1 \
  python src/main.py --configs \
  'configs/base.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_1gpu.jsonnet, configs/quantization/w8_sym_per_tensor.jsonnet' train