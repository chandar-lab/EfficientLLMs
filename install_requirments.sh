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
#module load gcc/9.3.0
module load python/3.8
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
pip install 'datasets==2.14.5'
pip install 'transformers==4.34'
pip install 'wandb==0.15.12'
pip install 'dill==0.3.4'
pip install 'scipy==1.8.0'
pip install 'matplotlib==3.7.0'
pip install 'accelerate==0.23.0'
pip install 'tiktoken==0.5.1'
pip install 'rouge_score>=0.1.2'
pip install 'tabulate'
pip install 'sacremoses'
pip install wheel setuptools py-cpuinfo

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
# there is an issue with installing layer_norm
# pushd ../layer_norm && pip install .
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

# save binary wheel to install on other machines
#TORCH_CUDA_ARCH_LIST="$NV_CC" DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 DS_BUILD_FUSED_LION=1 DS_BUILD_QUANTIZER=1 \
#python setup.py build_ext -j8 bdist_wheel
#pip install dist/*.whl --no-deps
#cp dist/*.whl $SCRATCH
