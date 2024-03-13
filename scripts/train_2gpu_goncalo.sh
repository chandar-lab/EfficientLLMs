#!/bin/bash
#SBATCH --job-name=gpt2
#SBATCH --time=4-00:00

#SBATCH --partition=main    # ask for unkillable job
#SBATCH --gpus-per-task=2               # number of gpus per node
#SBATCH --cpus-per-task=8              # number of cpus per gpu
#SBATCH --mem-per-gpu=16G               # memory per gpu
#SBATCH --ntasks=1
#SBATCH --constraint=80gb               # constraints
#SBATCH --exclude=cn-k[001-002]

export SLURM_CPUS_PER_GPU=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU

export CUBLAS_WORKSPACE_CONFIG=:16:8
export HF_HOME=$SCRATCH/hf_home
module load python/3.8
module load cuda/11.8
module load libffi
source ../ENV/bin/activate
torchrun --nproc_per_node=2 src/main.py --configs 'configs/base.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_2gpu_ddp.jsonnet, configs/quantization/g8_sym_per_tensor.jsonnet' train
#srun \
#  --nodes=$SLURM_JOB_NUM_NODES \
#  --ntasks=$SLURM_JOB_NUM_NODES \
#  --cpus-per-gpu=$SLURM_CPUS_PER_GPU \
#  --gpus-per-task=$SLURM_GPUS_PER_TASK \
#  --ntasks-per-node=1 \
#  python src/main.py --configs \
#  'configs/base.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/w4_sym_per_tensor.jsonnet' train