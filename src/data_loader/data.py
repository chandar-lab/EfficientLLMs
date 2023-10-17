import os
from common import FromParams, Registrable

from datasets import load_dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
from itertools import chain

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
                 add_eos: bool = True,
                 ):
        self.dataset_name = 'openwebtext'
        self.dataset_config_name = None
        self.tokenizer_name = tokenizer_name
        self.num_proc = num_proc
        self.max_length = max_length
        self.validation_size = validation_size
        self.val_split_seed = val_split_seed
        self.add_eos = add_eos

        if os.environ.get('HF_HOME') is not None:
            self.data_root = os.path.join(os.environ.get('HF_HOME'), 'datasets')
        else:
            hf_home = Path.home() / "scratch" / "hf_home"  # change this to your own path
            hf_home.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(hf_home)

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.padding_side = "right"

    # modified from https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    def creat_tokenized_datasets(self):
        raw_datasets = self._load_dataset()
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        if self.add_eos:
            add_eos = lambda seq: (seq + tokenizer.eos_token) if seq else seq
            add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
            tokenize = lambda example: tokenizer(add_eos_batched(example[text_column_name]))
        else:
            tokenize = lambda example: tokenizer(example[text_column_name])
        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32

        def tokenize_concat(examples):
            # We just need 'input_ids', not 'attention_mask' (since it's all 1)
            input_ids = np.fromiter(chain(*tokenize(examples)['input_ids']), dtype=dtype)
            # Need to return a list since we're doing batched processing
            return {'input_ids': [input_ids], 'len': [len(input_ids)]}

        tokenized_datasets = raw_datasets.map(
            tokenize_concat,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
        return tokenized_datasets

    def _load_dataset(self):
        raw_datasets = load_dataset(self.dataset_name, num_proc=self.num_proc)
        if 'validation' not in raw_datasets:
            assert "train" in raw_datasets, "You must have train in raw_datasets to make a validation raw_datasets"
            raw_datasets = raw_datasets["train"].train_test_split(
                test_size=self.validation_size, seed=self.val_split_seed,
                shuffle=True
            )
            raw_datasets['validation'] = raw_datasets['test']
        return raw_datasets

    def creat_data_collator(self):
        return DataCollatorForLanguageModeling(self.tokenizer, mlm=False)


@Dataset.register("openwebtext")
class OpenWebText(Dataset):
    def __init__(self,
                 tokenizer_name: str,
                 validation_size: float = 0.0005,
                 num_proc: int = 1,
                 max_length: int = 128,
                 train_batch_size: int = 2,
                 validation_batch_size: int = 2,
                 test_batch_size: int = 2,
                 val_split_seed: int = 2345,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 ):
        self.dataset_name = 'openwebtext'
        self.dataset_config_name = None
        self.tokenizer_name = tokenizer_name
        self.num_proc = num_proc
        self.max_length = max_length
        self.validation_size = validation_size

        self.train_batch_size = train_batch_size
        self.valid_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size

        self.shuffle = shuffle
        self.seed = 2345
        self.pin_memory = pin_memory

        if os.environ.get('HF_HOME') is not None:
            self.data_root = os.path.join(os.environ.get('HF_HOME'), 'datasets')
        else:
            hf_home = Path.home() / "scratch" / "hf_home"  # change this to your own path
            hf_home.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(hf_home)

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            # print("Add padding token")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.padding_side = "right"

    def creat_tokenized_datasets(self):

        def tokenize(element):
            outputs = self.tokenizer(
                element["text"],
                truncation=True,
                max_length=self.max_length,
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == self.max_length:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

        split_dataset = self._creat_dataset()
        tokenized_datasets = split_dataset.map(tokenize, batched=True, remove_columns=['text'],
                                               desc="tokenizing the splits", num_proc=self.num_proc)
        return tokenized_datasets

    def _creat_dataset(self, split: str = 'train'):
        dataset = load_dataset(self.dataset_name, num_proc=self.num_proc)
        # owt by default only contains the 'train' split, so create a test split
        split_dataset = dataset["train"].train_test_split(test_size=self.validation_size, seed=self.seed,
                                                          shuffle=self.shuffle)
        split_dataset["validation"] = split_dataset.pop('test')  # rename the test split to val
        return split_dataset

    def creat_data_collator(self):
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        return data_collator

    def build(self, split: str = 'train'):
        tokenized_datasets = self.creat_tokenized_datasets()
        data_collator = self.creat_data_collator()
        if split == 'train':
            dataloader = DataLoader(
                tokenized_datasets["train"], shuffle=self.shuffle, batch_size=self.train_batch_size,
                collate_fn=data_collator
            )
        elif split == 'val':
            dataloader = DataLoader(
                tokenized_datasets["validation"], batch_size=self.valid_batch_size, collate_fn=data_collator
            )
        else:
            raise "Didn't find split"
        return dataloader


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
