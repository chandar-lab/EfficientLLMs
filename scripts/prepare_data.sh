#!/bin/bash
#SBATCH --job-name=prepare-data-cpu
#SBATCH --time=0-3:00
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --ntasks=1

export CUBLAS_WORKSPACE_CONFIG=:16:8
export HF_HOME=$SCRATCH/hf_home
module load python/3.8
module load cuda/11.8
module load libffi
source ../ENV/bin/activate

python src/main.py --configs 'configs/llama_baseline.jsonnet' load_and_tokenize_dataset