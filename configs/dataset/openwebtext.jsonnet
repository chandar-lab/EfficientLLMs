local base = (import 'base.jsonnet');

base + {
    dataset+:{
        dataset_name: "openwebtext",
        dataset_config_name: null,
        tokenizer_name: "gpt2",
        max_length: 1024,
        train_batch_size: 8, # per gpu
        validation_batch_size: 4,
        num_workers: 4,
        shuffle: true,
        pin_memory: true,
        padding: 'max_length',
        },
}