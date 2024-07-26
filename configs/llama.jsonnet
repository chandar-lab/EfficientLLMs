local base_config = (import 'base.jsonnet');
local model_config = (import 'model/llama-small_2.jsonnet');
local dataset_config = (import 'dataset/openwebtext.jsonnet');
local train_config = (import 'trainer/gpt2_4gpu_ddp.jsonnet');

model_config + dataset_config + train_config + base_config