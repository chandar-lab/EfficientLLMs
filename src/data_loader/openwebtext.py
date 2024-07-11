import os
import transformers
from datasets import load_dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from transformers.testing_utils import CaptureLogger
from pathlib import Path
from itertools import chain
from data_loader import Dataset


@Dataset.register("openwebtext-hf")
class OpenWebText_HF(Dataset):
    def __init__(self,
                 tokenizer_name: str,
                 validation_size: float = 0.0005,
                 _num_proc: int = 1,
                 max_length: int = 128,
                 val_split_seed: int = 2345):
        """
        Initializes the OpenWebText_HF dataset class.

        Args:
            tokenizer_name (str): The name of the tokenizer to use.
            validation_size (float): The proportion of the dataset to use for validation.
            num_proc (int): Number of processes to use for data loading.
            max_length (int): Maximum sequence length for tokenization.
            val_split_seed (int): Seed for validation split.
        """
        self.dataset_name = 'openwebtext'
        self.dataset_config_name = None
        self.tokenizer_name = tokenizer_name
        self.num_proc = _num_proc
        self.max_length = max_length
        self.validation_size = validation_size
        self.val_split_seed = val_split_seed

        hf_home = Path.home() / "scratch" / "hf_home"  # Change this to your own path
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        self.data_root = os.path.join(os.environ.get('HF_HOME'), 'datasets')

        # Initialize tokenizer
        tokenizer_kwargs = {
            "cache_dir": None,
            "use_fast": True,
            "revision": "main",
            "token": None,
            "trust_remote_code": False,
            "add_prefix_space": True,
        }
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, **tokenizer_kwargs)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer = tokenizer
        self.save_directory = os.path.join(os.environ["HF_HOME"], f'datasets/openwebtext/tokenized_{self.tokenizer_name.replace("/", "-")}')

    # Modified from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
    def create_tokenized_datasets(self):
        """
        Creates and tokenizes the OpenWebText dataset.
        """
        raw_datasets = load_dataset(self.dataset_name, num_proc=self.num_proc)
        raw_datasets = raw_datasets["train"].train_test_split(test_size=self.validation_size, seed=self.val_split_seed, shuffle=True)
        raw_datasets["validation"] = raw_datasets.pop('test')  # Rename the test split to validation

        # Preprocessing the datasets
        column_names = list(raw_datasets["train"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        # Set up tokenizer logger
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = self.tokenizer(examples[text_column_name])
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
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

        block_size = self.max_length

        def group_texts(examples):
            # Concatenate all texts
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=self.num_proc,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        lm_datasets.save_to_disk(self.save_directory)
        print(f'Tokenized dataset saved at: {self.save_directory}')
        tok_logger.info(f'Tokenized dataset saved at: {self.save_directory}')

    def load_tokenized_dataset(self):
        """
        Loads the tokenized dataset from disk.

        Returns:
            DatasetDict: The tokenized dataset.
        """
        lm_datasets = load_from_disk(self.save_directory)
        return lm_datasets


if __name__ == "__main__":
    from common import Params

    dataset = Dataset.from_params(
        Params(
            {
                "type": "openwebtext-hf",
                "max_length": 1024,
                "num_proc": 4,
                "validation_size": 0.0005,
            }
        )
    )
    tokenizer = dataset.tokenizer
    tokenized_datasets = dataset.create_tokenized_datasets()
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    print(len(tokenized_datasets["train"]))
