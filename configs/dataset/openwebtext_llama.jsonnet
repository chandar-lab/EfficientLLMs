local base = (import 'base.jsonnet');

base + {
    dataset+:{
        type: "openwebtext-hf",
        tokenizer_name: "meta-llama/Llama-2-7b-hf",
        max_length: 1024,
        _num_proc: 24,
        val_split_seed: 2345,
        validation_size: 0.0005,
        },
}