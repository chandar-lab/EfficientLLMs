#!/bin/bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
module load libffi
source ../ENV/bin/activate
torchrun --nproc_per_node=8 src/main.py --configs 'configs/Mycfg.jsonnet, configs/quantization/4bit_wcat.jsonnet' train