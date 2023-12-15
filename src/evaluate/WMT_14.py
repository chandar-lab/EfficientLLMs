
from evaluate import Evaluate, compute_bleu
from tqdm import tqdm
from datasets import load_dataset
import torch
import warnings


@Evaluate.register("wmt14")
class WMT_14_Test(Evaluate):
    def __init__(self, task: str = 'fr-en', few_shot: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.raw_dataset = load_dataset("wmt14", task, num_proc=4, split='test')
        self.few_shot = few_shot

    def compute(self, model, tokenizer, stop=None):
        pass
