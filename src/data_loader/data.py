import os
from common import FromParams

from datasets import load_dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class Dataset(FromParams):
    def __init__(self,
                 dataset_name: str,
                 tokenizer_name: str,
                 dataset_config_name: str = None,
                 num_proc: int = 1,
                 padding: str = 'max_length',
                 max_length: int = 128,
                 train_batch_size: int = 2,
                 validation_batch_size: int = 2,
                 test_batch_size: int = 2,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 ):
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.num_proc = num_proc
        self.padding = padding
        self.max_length = max_length

        self.train_batch_size = train_batch_size
        self.valid_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size

        self.shuffle = shuffle
        self.pin_memory = pin_memory

        if os.environ.get('HF_HOME') is not None:
            self.data_root = os.path.join(os.environ.get('HF_HOME'), 'datasets')
        else:
            raise "didn't find HF HOME directory"

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

    def creat_tokenized_datasets(self):

        def tokenize(element):
            outputs = self.tokenizer(
                element["text"],
                truncation=True,
                max_length=self.max_length,
                padding=self.padding,  # Pad shorter sequences to the max_length
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = []
            for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
                if length == self.max_length:
                    input_batch.append(input_ids)
            return {"input_ids": input_batch}

        split_dataset = self._creat_dataset()
        tokenized_datasets = split_dataset.map(tokenize, batched=True,
                                               remove_columns=split_dataset["train"].column_names, num_proc=self.num_proc)
        return tokenized_datasets

    def _creat_dataset(self, split: str = 'train'):
        # split_dataset = load_dataset("wikitext", "wikitext-2-v1", num_proc=self.num_proc)
        split_dataset = load_dataset(self.dataset_name, self.dataset_config_name, num_proc=self.num_proc)
        return split_dataset

    def creat_data_collator(self):
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        return data_collator

    def build(self, split: str = 'train'):
        tokenized_datasets = self.creat_tokenized_datasets()
        data_collator = self.creat_data_collator()
        if split == 'train':
            dataloader = DataLoader(
                tokenized_datasets["train"], shuffle=self.shuffle, batch_size=self.train_batch_size, collate_fn=data_collator
            )
        elif split == 'val':
            dataloader = DataLoader(
                tokenized_datasets["validation"], batch_size=self.valid_batch_size, collate_fn=data_collator
            )
        elif split == 'test':
            dataloader = DataLoader(
                tokenized_datasets["test"], batch_size=self.test_batch_size, collate_fn=data_collator
            )
        else:
            raise "Didn't find split"
        return dataloader


if __name__ == "__main__":
    from common import Params

    dataset = Dataset.from_params(
        Params(
            {
                "dataset_name": "openwebtext",
                "dataset_config_name": None,
                "tokenizer_name": "gpt2",
                "max_length": 1024,
                "train_batch_size": 8,  # per gpu
                "validation_batch_size": 4,
                "num_workers": 4,
                "shuffle": True,
                "pin_memory": True,
                "padding": 'max_length',
            }
        )
    )
    tokenizer = dataset.tokenizer
    tokenized_datasets = dataset.creat_tokenized_datasets()
    data_collator = dataset.creat_data_collator()
    print(len(tokenized_datasets["train"]))
