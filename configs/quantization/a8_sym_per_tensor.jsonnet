local base = (import 'act_base.jsonnet');

base + {
    act_quantizer+:{
        N_bits: 8,
        signed:true,
        type: 'normal',
        granularity: 'per-tensor',
        inplace: false,
        all_positive: false,
        symmetric: true,
        minimum_range: 1e-5,
        beta: null,
        },
}