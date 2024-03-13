local base = (import 'base_adamw.jsonnet');

base + {
    optimizer+:{
        type: 'adamw',
        first_state_quantize_module:
            {
                N_bits: 4,
                signed:true,
                type: 'normal',
                granularity: 'per-group',
                inplace: false,
                all_positive: false,
                symmetric: true,
                minimum_range: 1e-8,
                beta: null,
                groupsize: 128,
            },
        },
}

