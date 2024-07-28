#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=eval
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00


export CUBLAS_WORKSPACE_CONFIG=:16:8
export HF_HOME=$SCRATCH/hf_home
module load python/3.8
module load cuda/11.8
module load libffi
source ../ENV/bin/activate

python src/evaluation.py --chk 60000 --config './configs/llama_baseline.jsonnet' --tasks 'all'

python src/evaluation.py --chk 60000 --config './configs/llama_baseline.jsonnet' --tasks 'ppl'