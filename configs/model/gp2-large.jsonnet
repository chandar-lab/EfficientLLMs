local base = (import 'gpt2-base.jsonnet');

base + {
    model+:{
        config+:{
            n_embd: 1280,
            n_head: 20,
            n_layer: 36,
        },
        },
}