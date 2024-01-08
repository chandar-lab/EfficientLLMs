local base = (import 'base.jsonnet');

base + {
    evaluation_task+:{
        type: 'cnn_daily_mail',
        task: '1.0.0',
        top_k: 100,
        top_p: 0.9,
        temperature: 1.0,
        max_new_tokens: 100,
        sentence_cuttoff: 3,
        hint: " TL;DR:",
        },
}
