local base = (import 'weight_base.jsonnet');

base + {
    weight_quantizer+:{
        N_bits: 8,
        get_scales+:{
            type: 'per-row'
        }
    },
}