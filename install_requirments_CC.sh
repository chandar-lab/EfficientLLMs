#!/bin/bash
set -euo pipefail

# Default config
if [ -z "${TMP_PYENV}" ]; then
    TMP_PYENV=$SCRATCH/ENV
fi

# Load modules
module load python/3.10 gcc/9.3.0 git-lfs/3.3.0 rust/1.70.0 protobuf/3.21.3 cuda/11.8.0 cudnn/8.6.0.163 arrow/12.0.1

# Create environment
python -m venv "$TMP_PYENV"
source "$TMP_PYENV/bin/activate"
pip install --upgrade pip

# download and compile python dependencies
pip uninstall -y ninja && pip install ninja
pip install 'packaging==23.1'
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# install flash-attention v2.3.2 from compiled file
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/flash_attn-2.3.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install 'jsonlines==3.0.0'
pip install 'jsonnet==0.18.0'
pip install 'overrides==6.1.0'
pip install 'fire==0.4.0'
pip install 'sentencepiece==0.1.96'
pip install 'nltk==3.6.7'
pip install 'base58==2.1.1'
pip install 'datasets==2.14.5'
pip install 'transformers==4.34'
pip install 'wandb==0.15.12'
pip install 'dill==0.3.7'
pip install 'scipy==1.11.2'
pip install 'matplotlib==3.7.2'
pip install 'accelerate<0.21.0,>=0.20.0'
pip install wheel setuptools py-cpuinfo

# Clone and install DeepSpeed
DeepSpeed_DIR="$WORK_DIR/deep_speed"
git clone https://github.com/microsoft/DeepSpeed/ "$DeepSpeed_DIR"
cd "$DeepSpeed_DIR"
git checkout "tags/v$DeepSpeed_VERSION"
rm -rf build
TORCH_CUDA_ARCH_LIST="$NV_CC" DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LION=1 DS_BUILD_QUANTIZER=1 \
pip install . --global-option="build_ext" --global-option="-j4" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
