from evaluate import Evaluate, compute_perplexity
from datasets import load_dataset
import torch


@Evaluate.register("1bw")
class OneBillionWord(Evaluate):
    def __init__(self, stride: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.raw_dataset = load_dataset("lm1b", num_proc=4, split='test')
        self.stride = stride
        self.task = 'ppl'

    @torch.inference_mode()
    def compute(self, model, tokenizer):
        return compute_perplexity(model, tokenizer, self.raw_dataset, self.stride)


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
                "type": "1bw",
                "stride": 512,
            }
        )
    )
    print(task.compute(model, tokenizer))
