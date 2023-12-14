local base = (import 'weight_base.jsonnet');

base + {
    weight_quantizer+:{
        N_bits: 8,
        signed:true,
        type: 'normal',
        granularity: 'per-column',
        inplace: false,
        all_positive: false,
        symmetric: true,
        minimum_range: 1e-5,
        beta: null,
        },
}