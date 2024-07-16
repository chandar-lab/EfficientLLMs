local base = (import 'base.jsonnet');
local deepspeed_config = (import '../deepspeed/zero_0.jsonnet');

base + {
    train_args+:{
        deepspeed: deepspeed_config,
        gradient_accumulation_steps: 2,
        per_device_eval_batch_size: 64,
        per_device_train_batch_size: 64,
        },
}

