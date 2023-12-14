#!/bin/bash
#SBATCH --account=rrg-franlp
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16000
#SBATCH --time=2:59:00

# Load modules
module load python/3.10 gcc/9.3.0 git-lfs/3.3.0 rust/1.70.0 protobuf/3.21.3 cuda/11.8.0 cudnn/8.6.0.163 arrow/12.0.1 httpproxy

# activate environment
source ../ENV/bin/activate

export HF_HOME=$SCRATCH/hf_home

# configure
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUBLAS_WORKSPACE_CONFIG=:16:8

python src/main.py --configs 'configs/base.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_1gpu.jsonnet' check_dataset