{
    optimizer: {
        weight_decay: 'auto',
        lr: 'auto',
        first_state_quantize_module: {
            type: 'linear',
            signed: true,
            get_scales:{
                symmetric: true,
                minimum_range: 1e-8,
            }
        },
        second_state_quantize_module: {
        type: 'linear',
            signed: true,
            get_scales:{
                symmetric: true,
                minimum_range: 1e-8,
            }
        },
    },
}