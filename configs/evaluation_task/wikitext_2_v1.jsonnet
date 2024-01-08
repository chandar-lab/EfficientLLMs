local base = (import 'base.jsonnet');

base + {
    evaluation_task+:{
        type: 'wikitext',
        task: 'wikitext-2-v1',
        stride: 64,
        },
}
