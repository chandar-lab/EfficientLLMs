local base = (import 'gpt2-base.jsonnet');

base + {
    model+:{
        config+:{
            n_embd: 1600,
            n_head: 25,
            n_layer: 48,
        },
        },
}