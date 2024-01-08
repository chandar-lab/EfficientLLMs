
from evaluate import Evaluate, compute_bleu
from tqdm import tqdm
from datasets import load_dataset
import torch
import warnings


@Evaluate.register("wmt14")
class WMT_14_Test(Evaluate):
    def __init__(self, task: str = 'fr-en', num_beams: int = 5, top_k: int = 1, top_p: float = 0.,
                 do_sample: bool = False, temperature: float = 1.0, few_shot: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.raw_dataset = load_dataset("wmt14", task, num_proc=4, split='test')
        self.num_beams = num_beams
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.temperature = temperature
        self.few_shot = few_shot

    def compute(self, model, tokenizer, stop=None):
        model = model.eval()
        # cumsum_cuda_kernel in flash attention does not support deterministic operation.
        torch.use_deterministic_algorithms(False)
        pass


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
                "type": "wmt14",
                "task": 'fr-en',
                "few_shot": 5,
            }
        )
    )
    print(task.compute(model, tokenizer))
