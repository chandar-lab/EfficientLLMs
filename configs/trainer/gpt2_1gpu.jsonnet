local base = (import 'base.jsonnet');

base + {
    train_args+:{
        gradient_accumulation_steps: 8,
        per_device_eval_batch_size: 64,
        per_device_train_batch_size: 64,
        },
}

