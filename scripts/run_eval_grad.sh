#!/bin/bash
#SBATCH --job-name=gpt2_1gpu
#SBATCH --time=0-03:00

#SBATCH --partition=main    # ask for unkillable job
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=2              # number of cpus per gpu
#SBATCH --mem-per-gpu=16G               # memory per gpu
#SBATCH --signal=TERM@60                # SIGTERM 60s prior to the allocation's end

export SLURM_CPUS_PER_GPU=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU

export CUBLAS_WORKSPACE_CONFIG=:16:8
export HF_HOME=$SCRATCH/hf_home
module load python/3.8
module load cuda/11.8
module load libffi
source ../ENV/bin/activate


python src/run_eval.py --tasks 'all' --num_fewshot 5 --provide_description --seed 42 1234 1337 2024 1357 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/g8_sym_per_column.jsonnet'

#python src/run_eval.py --tasks 'all' --num_fewshot 5 --provide_description --seed 42 1234 1337 2024 1357 \
#--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_2gpu_ddp.jsonnet, configs/quantization/g8_sym_per_tensor.jsonnet'

#python src/run_eval.py --tasks 'all' --num_fewshot 5 --provide_description --seed 42 1234 1337 2024 1357 \
#--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/g4_sym_per_column.jsonnet'
