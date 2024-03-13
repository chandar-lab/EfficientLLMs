local base = (import 'grad_base.jsonnet');

base + {
    grad_quantizer+:{
        N_bits: 4,
        signed:true,
        type: 'split_quant',
        granularity: 'per-column',
        inplace: false,
        all_positive: false,
        symmetric: true,
        minimum_range: 1e-9,
        beta: null,
        },
}