local base = (import 'base.jsonnet');

base + {
    evaluation_task+:{
        type: "wmt14",
        task: 'fr-en',
        few_shot: 5,
        },
}
