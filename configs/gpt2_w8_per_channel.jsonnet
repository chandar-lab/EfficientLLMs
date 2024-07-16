local baseline_config = (import 'gpt2_baseline.jsonnet');
local quantization_config = (import 'quantization/8bit_sym_per_channel.jsonnet');

baseline_config +{
    model+: {
    weight_quantize_module: quantization_config.quantizer,
  },
}
