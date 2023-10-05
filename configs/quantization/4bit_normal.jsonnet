local base = (import 'base.jsonnet');

base + {
    quantizer+:{
        N_bits: 4,
        signed:1,
        type: "normal",
        },
}