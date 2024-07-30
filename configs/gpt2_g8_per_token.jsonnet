local baseline_config = (import 'gpt2_baseline.jsonnet');
local quantization_config = (import 'quantization/8bit_sym_per_row.jsonnet') {
  quantizer+: {
    get_scales +: {
        minimum_range: 1e-9,
        }
  },
};

baseline_config +{
    model+: {
    grad_quantize_module: quantization_config.quantizer,
  },
}



