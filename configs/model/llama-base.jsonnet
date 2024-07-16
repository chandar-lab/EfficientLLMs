{
    model: {
    type: "llama",
    weight_quantize_module: {},
    act_quantize_module: {},
    grad_quantize_module: {},
    use_flash_attn: true,
        fused_bias_fc: false, # to be able to quantize wights of QKV projection
        fused_mlp: false, # to be able quantize weights of fc1 and fc2
        fused_dropout_add_ln: false, # fused dropout is not installed yet, due to some issues
        residual_in_fp32: true,
        pad_vocab_size_multiple: 8,
    },
}