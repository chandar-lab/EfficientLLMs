local base = (import 'gpt2-base.jsonnet');

base + {
    model+:{
        config+:{
            n_embd: 1024,
            n_head: 16,
            n_layer: 24,
        },
        },
}