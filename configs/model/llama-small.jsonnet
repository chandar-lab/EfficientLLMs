local base = (import 'llama-base.jsonnet');

base + {
    model+:{
        type: "llama",
        config+:{
            hidden_size: 640,
            num_hidden_layers: 24,
            num_attention_heads: 10,
            intermediate_size: 2560,
            vocab_size: 32000,
        },
        },
}