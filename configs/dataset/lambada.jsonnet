local base = (import 'base.jsonnet');

base + {
    dataset+:{
        type: "lambada",
        tokenizer_name: "gpt2",
        max_length: 1024,
        _num_proc: 24,
        },
}