from evaluate import Evaluate, compute_perplexity
from datasets import load_dataset
from transformers.testing_utils import CaptureLogger
import torch
from itertools import chain
from tqdm import tqdm
import os
import transformers
from transformers import Trainer
from transformers import TrainingArguments

@Evaluate.register("openwebtext")
class OpenWebText_Evaluation(Evaluate):
    def __init__(self, task: str = 'loss', stride: int = 512,
                 batch_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        raw_datasets = load_dataset("openwebtext", num_proc=4)
        raw_datasets = raw_datasets["train"].train_test_split(test_size=0.0005, seed=2345, shuffle=True)
        self.raw_datasets = raw_datasets.pop('test')
        self.stride = stride
        self.task = task
        self.batch_size = batch_size

    @torch.inference_mode()
    def compute(self, model, tokenizer):
        model = model.eval()
        if self.task == 'ppl':
            return compute_perplexity(model, tokenizer, self.raw_dataset, self.stride)
        elif self.task == 'loss':
            lm_dataset = self.prep_dataset(tokenizer, model.config.n_positions)
            # loss = 0
            # for inputs in tqdm(lm_dataset):
            #     input_ids = torch.tensor(inputs['input_ids'])[None, :].to('cuda')
            #     target_ids = input_ids.clone()
            #     outputs = model(input_ids, labels=target_ids)
            #     loss += outputs.loss.item()
            # return loss
            os.environ["WANDB_DISABLED"] = "true"
            train_args = TrainingArguments(output_dir='./save_test', report_to=None, per_device_eval_batch_size=self.batch_size)
            trainer = Trainer(model=model, args=train_args, eval_dataset=lm_dataset)
            metrics = trainer.evaluate()
            return metrics

        else:
            raise NotImplementedError

    def prep_dataset(self, tokenizer, n_positions=1024):
        block_size = n_positions

        column_names = list(self.raw_datasets.features)
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

        tokenized_datasets = self.raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=column_names,
            load_from_cache_file=True,  # default value of data_args.overwrite_cache is False
            desc="Running tokenizer on dataset",
        )

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

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        return lm_datasets
