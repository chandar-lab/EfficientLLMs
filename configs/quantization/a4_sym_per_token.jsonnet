local base = (import 'act_base.jsonnet');

base + {
    act_quantizer+:{
        N_bits: 4,
        signed:true,
        type: 'normal',
        granularity: 'per-token',
        inplace: false,
        all_positive: false,
        symmetric: true,
        minimum_range: 1e-5,
        beta: null,
        },
}