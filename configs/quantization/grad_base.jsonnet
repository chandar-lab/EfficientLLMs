{
    grad_quantizer: {
        type: 'split_quant',
        signed: true,
        get_scales:{
            symmetric: true,
            minimum_range: 1e-9,
        }
    },
}