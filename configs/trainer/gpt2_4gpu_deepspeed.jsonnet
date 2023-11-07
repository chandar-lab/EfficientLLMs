local base = (import 'base.jsonnet');
local deepspeed_config = (import '../deepspeed/zero_0.jsonnet');

base + {
    train_args+:{
        fp16: false,
        bf16: true,
        bf16_full_eval: true,
        dataloader_drop_last: true,
        dataloader_num_workers: 4,
        dataloader_pin_memory: true,
        deepspeed: deepspeed_config,
        do_eval: true,
        do_train: true,
        evaluation_strategy: "steps",
        eval_steps: 100,
        gradient_accumulation_steps: 2,
        gradient_checkpointing: false,
        learning_rate: 6e-4,
        log_on_each_node: false,
        logging_steps: 1,
        max_grad_norm: 1.0,
        max_steps: 300000,
        per_device_eval_batch_size: 64,
        per_device_train_batch_size: 64,
        save_steps: 1000,
        save_strategy: "steps",
        save_total_limit: 100,
        torch_compile: false,
        warmup_ratio: 0.0,
        weight_decay: 0.1,
        },
}

