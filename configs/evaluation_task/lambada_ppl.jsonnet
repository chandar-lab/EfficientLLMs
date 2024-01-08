local base = (import 'base.jsonnet');

base + {
    evaluation_task+:{
        type: 'lambada',
        task: 'ppl',
        stride: 64,
        },
}
