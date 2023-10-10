local base = (import 'base.jsonnet');

base + {
    dataset+:{
        type: "openwebtext",
        tokenizer_name: "gpt2",
        max_length: 1024,
        train_batch_size: 8, # per gpu
        validation_batch_size: 4,
        num_proc: 4,
        shuffle: true,
        pin_memory: true,
        validation_size: 0.0005,
        },
}