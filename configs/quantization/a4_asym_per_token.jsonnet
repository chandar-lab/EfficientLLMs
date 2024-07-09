local base = (import 'act_base.jsonnet');

base + {
    act_quantizer+:{
        N_bits: 4,
        get_scales+:{
            symmetric: false,
            type: 'per-row',
        }
    },
}