local base = (import 'grad_base.jsonnet');

base + {
    grad_quantizer+:{
        N_bits: 4,
        get_scales+:{
            type: 'per-tensor'
        }
    },
}