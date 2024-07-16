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
DeepSpeed_VERSION='0.11.1'
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
module load python/3.8
module load cuda/11.8

# Create environment
python -m venv "$TMP_PYENV"
source "$TMP_PYENV/bin/activate"
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Note: Ensure that the installed PyTorch version supports CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Note: Ensure that ninja is uninstalled and reinstalled to avoid conflicts
pip uninstall -y ninja && pip install ninja

# Clone and install flash-attention v2
NV_CC="8.0;8.6" # flash-attention-v2 and exllama_kernels are anyway limited to CC of 8.0+
FLASH_ATTENTION_DIR="$WORK_DIR/flash-attention-v2"
git clone https://github.com/Dao-AILab/flash-attention "$FLASH_ATTENTION_DIR"
pushd "$FLASH_ATTENTION_DIR"
git checkout "tags/v$FLASH_ATTN_VERSION"
TORCH_CUDA_ARCH_LIST="$NV_CC" MAX_JOBS="$MAX_JOBS" python setup.py install
pushd csrc/fused_dense_lib && pip install .
pushd ../xentropy && pip install .
pushd ../rotary && pip install .
pushd ../layer_norm && pip install .
popd  # Exit from csrc/rotary
popd  # Exit from flash-attention


# Clone and install DeepSpeed
DeepSpeed_DIR="$WORK_DIR/deep_speed"
git clone https://github.com/microsoft/DeepSpeed/ "$DeepSpeed_DIR"
cd "$DeepSpeed_DIR"
git checkout "tags/v$DeepSpeed_VERSION"
rm -rf build
TORCH_CUDA_ARCH_LIST="$NV_CC" DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LION=1 DS_BUILD_QUANTIZER=1 \
pip install . --global-option="build_ext" --global-option="-j4" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log