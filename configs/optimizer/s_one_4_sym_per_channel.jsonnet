local base = (import 'base_adamw.jsonnet');

base + {
    optimizer+:{
        type: 'adamw',
        first_state_quantize_module:
            {
                N_bits: 4,
                get_scales+:{
                    type: 'per-row'
                }
            },
        },
}

