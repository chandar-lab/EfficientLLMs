#!/bin/bash
#SBATCH --partition=main
#SBATCH -J install-requirements
#SBATCH --output=%x.%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=6:00:00
set -e
set -v

FLASH_ATTN_VERSION='2.3.2'
export MAX_JOBS=4

# Default config
if [ -z "${TMP_PYENV}" ]; then
    TMP_PYENV=$SCRATCH/ENV
fi
if [ -z "${WORK_DIR}" ]; then
    WORK_DIR=$SLURM_TMPDIR/workspace
fi
mkdir -p $WORK_DIR

# Load modules
module load gcc/9.3.0
module load python/3.10
module load cuda/11.8

# Create environment
python -m venv "$TMP_PYENV"
source "$TMP_PYENV/bin/activate"
pip install --upgrade pip

# download and compile python dependencies
pip uninstall -y ninja && pip install ninja
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install 'jsonlines==3.0.0'
pip install 'jsonnet==0.18.0'
pip install 'packaging==23.2'
pip install 'overrides==6.1.0'
pip install 'fire==0.4.0'
pip install 'sentencepiece==0.1.96'
pip install 'nltk==3.6.7'
pip install 'base58==2.1.1'
pip install 'datasets==1.18.2'
pip install 'transformers==4.16.2'
pip install 'wandb==0.12.10'
pip install 'dill==0.3.4'
pip install 'scipy==1.8.0'
pip install 'matplotlib==3.8.0'
pip install 'accelerate<0.21.0,>=0.20.0'
pip install wheel
### tiktoken PyYAML

# Clone and install flash-attention v2
NV_CC="8.0;8.6" # flash-attention-v2 and exllama_kernels are anyway limited to CC of 8.0+
FLASH_ATTENTION_DIR="$WORK_DIR/flash-attention-v2"
git clone https://github.com/Dao-AILab/flash-attention "$FLASH_ATTENTION_DIR"
pushd "$FLASH_ATTENTION_DIR"
git checkout "tags/v$FLASH_ATTN_VERSION"
TORCH_CUDA_ARCH_LIST="$NV_CC" MAX_JOBS="$MAX_JOBS" python setup.py install
pushd csrc/fused_dense_lib && pip install .
pushd csrc/xentropy && pip install .
pushd csrc/rotary && pip install .
# there is an issue with installing layer_norm
# cd pushd csrc/layer_norm && pip install .
popd  # Exit from csrc/rotary

popd  # Exit from flash-attention
