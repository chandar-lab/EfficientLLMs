{
    act_quantizer: {
        type: 'linear',
        signed: true,
        get_scales:{
            symmetric: true,
            minimum_range: 1e-5,
        }
    },
}