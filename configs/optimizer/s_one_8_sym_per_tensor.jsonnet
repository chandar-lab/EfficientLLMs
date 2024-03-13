local base = (import 'base_adamw.jsonnet');

base + {
    optimizer+:{
        type: 'adamw',
        first_state_quantize_module:
            {
                N_bits: 8,
                signed:true,
                type: 'normal',
                granularity: 'per-tensor',
                inplace: false,
                all_positive: false,
                symmetric: true,
                minimum_range: 1e-8,
                beta: null,
            },
        },
}

