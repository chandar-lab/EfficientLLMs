local base = (import 'act_base.jsonnet');

base + {
    act_quantizer+:{
        N_bits: 8,
        get_scales+:{
            type: 'per-tensor',
        }
    },
}