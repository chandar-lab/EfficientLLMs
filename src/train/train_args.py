from common import FromParams
from transformers import TrainingArguments

class MyTrainingArguments(TrainingArguments, FromParams):
    def __init__(self, output_dir='./save', data_seed=42, seed=42,
                 run_name='test', report_to=None, **kwargs):
        super().__init__(output_dir=output_dir, data_seed=data_seed, seed=seed,
                         logging_dir=output_dir, run_name=run_name,
                         report_to=report_to, **kwargs)
