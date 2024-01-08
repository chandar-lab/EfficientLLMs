from common import FromParams
from transformers import TrainingArguments

class MyTrainingArguments(TrainingArguments, FromParams):
    def __init__(self, output_dir='./save', data_seed=42, seed=42,
                 run_name='test', report_to=None, save_spec_steps=None,  **kwargs):
        if 'deepspeed' in kwargs.keys():
            deepspeed = kwargs.pop('deepspeed')
            deepspeed = deepspeed.as_dict()
        else:
            deepspeed = None
        super().__init__(output_dir=output_dir, data_seed=data_seed, seed=seed,
                         logging_dir=output_dir, run_name=run_name,
                         report_to=report_to, deepspeed=deepspeed, **kwargs)
        self.save_spec_steps = save_spec_steps
