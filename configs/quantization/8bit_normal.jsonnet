local base = (import 'base.jsonnet');

base + {
    quantizer+:{
        N_bits: 8,
        signed:true,
        granularity: 'per-column',
        type: "normal",
        },
}