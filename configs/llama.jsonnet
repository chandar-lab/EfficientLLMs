local base_config = (import 'base.jsonnet');
local model_config = (import 'model/llama-small.jsonnet');
local dataset_config = (import 'dataset/openwebtext_llama.jsonnet');
local train_config = (import 'trainer/gpt2_1gpu.jsonnet');

model_config + dataset_config + train_config + base_config
