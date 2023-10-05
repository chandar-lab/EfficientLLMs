from common import FromParams
from transformers import TrainingArguments

class MyTrainingArguments(TrainingArguments, FromParams):
    def __init__(self, output_dir='./save', **kwargs):
        super().__init__(output_dir=output_dir, **kwargs)
