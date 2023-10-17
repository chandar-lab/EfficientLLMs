#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=8:00:00

export CUBLAS_WORKSPACE_CONFIG=:16:8
module load libffi
source ../ENV/bin/activate
python src/main.py --configs 'configs/Mycfg.jsonnet' train