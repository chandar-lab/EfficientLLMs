local base = (import 'base_adamw.jsonnet');
local quantize_config = (import '../quantization/w8_sym_per_column.jsonnet');

base + {
    optimizer+:{
        type: 'adamw',
        grad_quantize_module: quantize_config.weight_quantizer
        },
}

