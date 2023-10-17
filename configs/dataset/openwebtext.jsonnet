local base = (import 'base.jsonnet');

base + {
    dataset+:{
        type: "openwebtext-hf",
        tokenizer_name: "gpt2",
        max_length: 1024,
        num_proc: 4,
        val_split_seed: 2345,
        validation_size: 0.0005,
        add_eos: true,
        },
}