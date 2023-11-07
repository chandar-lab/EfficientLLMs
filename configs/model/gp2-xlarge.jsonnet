local base = (import 'base.jsonnet');

base + {
    model+:{
        type: "gpt2",
        use_flash_attn: true,
        fused_bias_fc: false, # to be able to quantize wights of QKV projection
        fused_mlp: false, # to be able quantize weights of fc1 and fc2
        fused_dropout_add_ln: false, # fused dropout is not installed yet, due to some issues
        residual_in_fp32: true,
        pad_vocab_size_multiple: 8,
        weight_quantize_module: {},
        config: {
            activation_function: "gelu_new",
            attn_pdrop: 0.1,
            bos_token_id: 50256,
            embd_pdrop: 0.1,
            eos_token_id: 50256,
            initializer_range: 0.02,
            layer_norm_epsilon: 1e-05,
            model_type: "gpt2",
            n_embd: 1600,
            n_head: 25,
            n_inner: null,
            n_layer: 48,
            n_positions: 1024,
            reorder_and_upcast_attn: false,
            resid_pdrop: 0.1,
            scale_attn_by_inverse_layer_idx: false,
            scale_attn_weights: true,
            summary_activation: null,
            summary_first_dropout: 0.1,
            summary_proj_to_labels: true,
            summary_type: "cls_index",
            summary_use_proj: true,
            transformers_version: "4.33.1",
            use_cache: true,
            vocab_size: 50257
        },
        },
}