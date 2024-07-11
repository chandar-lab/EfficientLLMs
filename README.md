# Exploring Quantization for Efficient Pre-Training of Transformer Language Models

This repository contains the code for Exploring Quantization for Efficient Pre-Training of Transformer Language Models.

## Abstract
The increasing scale of Transformer models has led to an increase in their pre-training computational requirements. While quantization has proven to be effective after pre-training and during fine-tuning, applying quantization in Transformers during pre-training has remained largely unexplored at scale for language modeling. This study aims to explore the impact of quantization for efficient pre-training of Transformers, with a focus on linear layer components. By systematically applying straightforward linear quantization to weights, activations, gradients, and optimizer states, we assess its effects on model efficiency, stability, and performance during training. By offering a comprehensive recipe of effective quantization strategies to be applied during the pre-training of Transformers, we promote high training efficiency from scratch while retaining language modeling ability.

## Installation
Python: 3.8+ , CUDA: 11.8
1. Clone the repository.
2. Create a virtual environment and activate it:

```bash
python -m venv env
source env/bin/activate
```

3. Install the dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Follow additional steps to install FlashAttention as shown in the `scripts/install_requirements.sh` script.

## Running experiments

Experiments were conducted using the OpenWebText dataset from [HuggingFace](https://huggingface.co/datasets/Skylion007/openwebtext), following a set of training configurations similar to those in [nanoGPT](https://github.com/karpathy/nanoGPT). These experiments were run on 4xA100 80G GPUs.

### Prepare dataset 

To download and tokenize the dataset into your `$HF_HOME` directory, run:
```
python src/main.py --configs 'configs/gpt2_baseline.jsonnet' load_and_tokenize_datase
```

### Training

Our training utilizes the HuggingFace Trainer. You can adjust the training configurations in the `configs/trainer` directory. On average, training takes approximately 4.3 days to complete 300k steps.
```
torchrun --nproc_per_node=4 src/main.py --configs 'configs/gpt2_baseline.jsonnet' train
```
We provide scripts for running the experiments to a Slurm queue in `scripts/train.sh`

### Evaluation

To evaluate the model's performance on a selected task, using the same configuration for the model, training recipe, dataset, and quantization, run:
```
python src/main.py --configs 'configs/gpt2_baseline.jsonnet, configs/evaluation_task/hellaswag.jsonnet' evaluate
```


## Citation