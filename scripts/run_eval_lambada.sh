#!/bin/bash
#SBATCH --job-name=gpt2_1gpu
#SBATCH --time=0-03:00

#SBATCH --partition=main    # ask for unkillable job
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=2              # number of cpus per gpu
#SBATCH --mem-per-gpu=16G               # memory per gpu
#SBATCH --signal=TERM@60                # SIGTERM 60s prior to the allocation's end

export SLURM_CPUS_PER_GPU=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU

export CUBLAS_WORKSPACE_CONFIG=:16:8
export HF_HOME=$SCRATCH/hf_home
module load python/3.8
module load cuda/11.8
module load libffi
source ../ENV/bin/activate


python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/w8_sym_per_column.jsonnet, configs/quantization/a8_sym_per_column.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp_CC.jsonnet, configs/quantization/w8_sym_per_column.jsonnet, configs/quantization/a8_sym_per_column.jsonnet, configs/quantization/g8_sym_per_column.jsonnet'

### weight

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/w8_sym_per_column.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/w8_sym_per_tensor.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/w4_sym_per_column.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/w4_sym_per_tensor.jsonnet'

### activation

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/a8_sym_per_column.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/a8_sym_per_tensor.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/a4_sym_per_column.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp_CC.jsonnet, configs/quantization/a4_sym_per_token.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/a4_asym_per_column.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/a4_sym_per_tensor.jsonnet'

### grads

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/g8_sym_per_column.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_2gpu_ddp.jsonnet, configs/quantization/g8_sym_per_tensor.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/quantization/g4_sym_per_column.jsonnet'

### states

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp_CC.jsonnet, configs/optimizer/s_one_8_sym_per_column.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp_CC.jsonnet, configs/optimizer/s_one_8_sym_per_tensor.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet, configs/optimizer/s_one_4_sym_per_column.jsonnet'

python src/run_eval.py --tasks 'lambada' --output_path 'LAMBADA' --num_fewshot 1 --provide_description --seed 1234 \
--configs 'configs/base_new.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp_CC.jsonnet, configs/optimizer/s_one_4_sym_per_tensor.jsonnet'