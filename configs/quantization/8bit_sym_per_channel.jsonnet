local base = (import 'base.jsonnet');

base + {
    quantizer+:{
        N_bits: 8,
        get_scales+:{
            type: 'per-column',
        }
    },
}