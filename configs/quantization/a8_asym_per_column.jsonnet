local base = (import 'act_base.jsonnet');

base + {
    act_quantizer+:{
        N_bits: 8,
        signed:true,
        type: 'normal',
        granularity: 'per-column',
        inplace: false,
        all_positive: false,
        symmetric: false,
        minimum_range: 1e-5,
        beta: null,
        },
}