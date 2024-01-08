# code modified from: https://github.com/cybertronai/bflm/blob/master/eval_lambada_slow.py

from evaluate import Evaluate, get_rouge_score
from tqdm import tqdm
from datasets import load_dataset
import torch
import warnings


@Evaluate.register("cnn_daily_mail")
class CNN_Daily_Mail(Evaluate):
    def __init__(self, task: str = '1.0.0', top_k: int = 1, top_p: float = 0., temperature: float = 1.0,
                 max_new_tokens: int = 100, sentence_cuttoff: int = 3, hint: str = " TL;DR:", **kwargs):
        super().__init__(**kwargs)
        self.raw_dataset = load_dataset("cnn_dailymail", task, num_proc=4, split='test')
        self.max_new_tokens = max_new_tokens
        self.sentence_cuttoff = sentence_cuttoff
        self.hint = hint
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    def compute(self, model, tokenizer, stop=None):
        model = model.eval()
        # cumsum_cuda_kernel in flash attention does not support deterministic operation.
        torch.use_deterministic_algorithms(False)
        model.config.pad_token_id = model.config.eos_token_id  # to avoid warning
        rougeLsum = 0
        for idx, sample in tqdm(enumerate(self.raw_dataset), total=len(self.raw_dataset)):
            if stop is not None and idx > stop:
                break

            prompt = sample['article'] + self.hint
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]
            if input_length > model.config.n_ctx - self.max_new_tokens:
                warnings.warn('contex larger than 1024 is not supported!')
                continue

            generated_tokens = torch.zeros(0, 1)
            while generated_tokens.shape[1]<self.max_new_tokens:
                # outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                #                          max_length=self.max_new_tokens + input_length, top_k=self.top_k,
                #                          do_sample=True)
                outputs = model.generate(inputs.input_ids, max_length=self.max_new_tokens + input_length,
                                         top_k=self.top_k, temperature=self.temperature)
                generated_tokens = outputs[:, input_length:]

            summarized = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            sentences = summarized.split('.')
            first_sentences = '.'.join(sentences[:self.sentence_cuttoff]) + '.'
            result = get_rouge_score(sample['article'], first_sentences, tokenizer)
            rougeLsum += result['rougeLsum']

        return rougeLsum / idx


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from transformers import GPT2LMHeadModel
    from evaluate import Evaluate
    from common import Params
    import torch

    tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": "main",
        "token": None,
        "trust_remote_code": False,
        "add_prefix_space": True
    }
    tokenizer = AutoTokenizer.from_pretrained("gpt2", **tokenizer_kwargs)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dtype=torch.bfloat16, device='cuda').eval()

    task = Evaluate.from_params(
        Params(
            {
                "type": "cnn_daily_mail",
                "task": '1.0.0',
                "top_k": 100,
                "top_p": 0.9,
                "temperature": 1.0,
                "max_new_tokens": 100,
                "sentence_cuttoff": 3,
                "hint": " TL;DR:",
            }
        )
    )
    print(task.compute(model, tokenizer))
