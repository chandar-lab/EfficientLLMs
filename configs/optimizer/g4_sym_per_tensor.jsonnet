local base = (import 'base_adamw.jsonnet');
local quantize_config = (import '../quantization/w4_sym_per_tensor.jsonnet');

base + {
    optimizer+:{
        type: 'adamw',
        grad_quantize_module: quantize_config.weight_quantizer
        },
}

