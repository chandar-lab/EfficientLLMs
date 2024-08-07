local baseline_config = (import 'gpt2_baseline.jsonnet');
local quantization_config_w = (import 'quantization/8bit_sym_per_row.jsonnet');
local quantization_config_a = (import 'quantization/8bit_sym_per_row.jsonnet');
local quantization_config_g = (import 'quantization/8bit_sym_per_row.jsonnet') {
  quantizer+: {
    get_scales +: {
        minimum_range: 1e-9,
        }
  },
};

baseline_config +{
    model+:{
    weight_quantize_module: quantization_config_w.quantizer,
    act_quantize_module: quantization_config_a.quantizer,
    grad_quantize_module: quantization_config_g.quantizer,
  },
}
