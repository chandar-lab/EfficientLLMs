local base = (import 'gpt2-base.jsonnet');

base + {
    model+:{
        config+:{
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
        },
        },
}