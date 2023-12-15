from evaluate import Evaluate, compute_perplexity
from datasets import load_dataset
import torch


@Evaluate.register("wikitext")
class WikiText(Evaluate):
    def __init__(self, task: str = 'wikitext-2-v1', stride: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.raw_dataset = load_dataset("wikitext", task, num_proc=4, split='test')
        self.stride = stride

    @torch.inference_mode()
    def compute(self, model, tokenizer):

        return compute_perplexity(model, tokenizer, self.raw_dataset, self.stride)


