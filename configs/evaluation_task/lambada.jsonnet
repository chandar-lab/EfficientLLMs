local base = (import 'base.jsonnet');

base + {
    evaluation_task+:{
        type: 'lambada',
        detokenize: true,
        detokenize_havent: true,
        stop_word_filter: false,
        num_beams: 5,
        top_k: 50,
        top_p: 0.95,
        do_sample: true,
        temperature: 1,
        },
}
