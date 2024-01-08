local base = (import 'base.jsonnet');

base + {
    evaluation_task+:{
        type: 'lambada',
        task: 'accuracy',
        detokenize: true,
        stop_word_filter: true,
        detokenize_havent: false,
        },
}

