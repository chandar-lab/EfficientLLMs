local baseline_config = (import 'llama_baseline.jsonnet');
local quantization_config_w = (import 'quantization/8bit_sym_per_row.jsonnet');
local quantization_config_a = (import 'quantization/8bit_sym_per_row.jsonnet');

baseline_config +{
    model+: {
    weight_quantize_module: quantization_config_w.quantizer,
    act_quantize_module: quantization_config_a.quantizer,
  },
}
