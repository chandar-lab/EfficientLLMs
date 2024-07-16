local base = (import 'base.jsonnet');

base + {
    evaluation_task+:{
        type: 'hellaswag',
        num_fewshot: 5,
        provide_description: true,
        },
}
