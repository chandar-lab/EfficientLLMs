local base = (import 'base.jsonnet');

base + {
    evaluation_task+:{
        type: 'wikitext_2',
        num_fewshot: 0,
        provide_description: false,
        stride: 1024,
        },
}