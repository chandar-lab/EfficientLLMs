import os
from common import FromParams, Registrable

import transformers
from datasets import load_dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from transformers.testing_utils import CaptureLogger
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from itertools import chain
import logging
from data_loader import Dataset

@Dataset.register("lambada")
class LAMBADA(Dataset):
    def __init__(self,
                 tokenizer_name: str,
                 _num_proc: int = 1,
                 max_length: int = 128,
                 ):
        self.dataset_name = 'lambada'
        self.dataset_config_name = None
        self.tokenizer_name = tokenizer_name
        self.num_proc = _num_proc
        self.max_length = max_length

        hf_home = Path.home() / "scratch" / "hf_home"  # change this to your own path
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        self.data_root = os.path.join(os.environ.get('HF_HOME'), 'datasets')

    # modified from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
    def creat_tokenized_datasets(self):
        raw_datasets = load_dataset("lambada", num_proc=self.num_proc)

        # default args
        tokenizer_kwargs = {
            "cache_dir": None,
            "use_fast": True,
            "revision": "main",
            "token": None,
            "trust_remote_code": False,
        }
        tokenizer = AutoTokenizer.from_pretrained("gpt2", **tokenizer_kwargs)

        # Preprocessing the datasets.
        column_names = list(raw_datasets["train"].features)
        # column_names = list(raw_datasets["validation"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return output

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=column_names,
            load_from_cache_file=True,  # default value of data_args.overwrite_cache is False
            desc="Running tokenizer on dataset",
        )

        return tokenized_datasets


if __name__ == "__main__":
    from common import Params

    dataset = Dataset.from_params(
        Params(
            {
                "type": "lambda",
                "tokenizer_name": "gpt2",
                "max_length": 1024,
                "num_proc": 4,
            }
        )
    )
    tokenizer = dataset.tokenizer
    tokenized_datasets = dataset.creat_tokenized_datasets()
    data_collator = dataset.creat_data_collator()
    print(len(tokenized_datasets["train"]))
