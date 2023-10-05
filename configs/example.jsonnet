local base_config = (import 'base.jsonnet');
local model_config = (import 'model/gp2-small.jsonnet');
local quantization_config = (import 'quantization/4bit_normal.jsonnet');
local dataset_config = (import 'dataset/openwebtext.jsonnet');
local train_config = (import 'trainer/gpt2_train_args.jsonnet');

model_config + {
  model+: {
    weight_quantize_module: quantization_config.quantizer,
  },
} + dataset_config + train_config + base_config