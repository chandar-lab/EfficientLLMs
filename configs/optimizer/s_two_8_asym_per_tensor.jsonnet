local base = (import 'base_adamw.jsonnet');

base + {
    optimizer+:{
        type: 'adamw',
        second_state_quantize_module:
            {
                N_bits: 8,
                signed:true,
                type: 'normal',
                granularity: 'per-tensor',
                inplace: false,
                all_positive: true,
                symmetric: true,
                minimum_range: 1e-8,
                beta: null,
            },
        },
}

