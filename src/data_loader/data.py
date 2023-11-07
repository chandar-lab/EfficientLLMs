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

class Dataset(Registrable):
    pass


@Dataset.register("openwebtext-hf")
class OpenWebText_HF(Dataset):
    def __init__(self,
                 tokenizer_name: str,
                 validation_size: float = 0.0005,
                 num_proc: int = 1,
                 max_length: int = 128,
                 val_split_seed: int = 2345,
                 ):
        self.dataset_name = 'openwebtext'
        self.dataset_config_name = None
        self.tokenizer_name = tokenizer_name
        self.num_proc = num_proc
        self.max_length = max_length
        self.validation_size = validation_size
        self.val_split_seed = val_split_seed

        hf_home = Path.home() / "scratch" / "hf_home"  # change this to your own path
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        self.data_root = os.path.join(os.environ.get('HF_HOME'), 'datasets')

    # modified from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
    def creat_tokenized_datasets(self):
        raw_datasets = load_dataset("openwebtext", num_proc=self.num_proc)
        raw_datasets = raw_datasets["train"].train_test_split(test_size=self.validation_size, seed=self.val_split_seed, shuffle=True)
        raw_datasets["validation"] = raw_datasets.pop('test')  # rename the test split to val

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
        block_size = self.max_length
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=self.num_proc,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        return lm_datasets


if __name__ == "__main__":
    from common import Params

    dataset = Dataset.from_params(
        Params(
            {
                "type": "openwebtext",
                "tokenizer_name": "gpt2",
                "max_length": 1024,
                "train_batch_size": 8,  # per gpu
                "validation_batch_size": 4,
                "num_proc": 4,
                "shuffle": True,
                "pin_memory": True,
                "validation_size": 0.0005,
            }
        )
    )
    tokenizer = dataset.tokenizer
    tokenized_datasets = dataset.creat_tokenized_datasets()
    data_collator = dataset.creat_data_collator()
    print(len(tokenized_datasets["train"]))
