#!/bin/bash
#SBATCH --account=rrg-franlp
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=492G
#SBATCH --job-name=gpt2_deepspeed
#SBATCH --time=0-03:00

# Load modules
module load python/3.10 gcc/9.3.0 git-lfs/3.3.0 rust/1.70.0 protobuf/3.21.3 cuda/11.8.0 cudnn/8.6.0.163 arrow/12.0.1 httpproxy

# activate environment
source ../ENV/bin/activate

CURRENT_HF_HOME=$SCRATCH/hf_home
export HF_HOME=$SLURM_TMPDIR/hf_home
cp -r $CURRENT_HF_HOME $HF_HOME

# configure
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUBLAS_WORKSPACE_CONFIG=:16:8

torchrun --nproc_per_node=4 src/main.py --configs 'configs/base.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_deepspeed_CC.jsonnet' train