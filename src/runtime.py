import os
import random
import numpy as np
import torch
import logging

import json
from models import Base_Model
from data_loader import Dataset
from common import FromParams, Lazy, Params
from typing import Dict, Any

from train import MyTrainingArguments
from train import MyHFTrainer
from callbacks import WandbCallback
import transformers
import datasets
from optimizer import Base_Optimizer
from lm_eval.tasks import HFTask
from accelerate import Accelerator

logger = logging.getLogger(__name__)

class Runtime(FromParams):
    def __init__(self,
                 model: Lazy[Base_Model],
                 evaluation_task: Lazy[HFTask],
                 train_args: Lazy[MyTrainingArguments],
                 dataset: Lazy[Dataset],
                 optimizer: Lazy[Base_Optimizer],
                 seed: int,
                 exp_name: str,
                 _project_name: str,
                 _entity: str,
                 _wandb_logs: bool = False,
                 _save_path: str = './save',
                 _config_copy: Dict[str, Any] = None):
        # setup model
        self.model = model.construct()
        # self.model.to(self.model.device)
        self.train_args = train_args
        self.trainer = None
        self.dataset = dataset
        if len(evaluation_task._params) > 0:
            self.evaluation_task = evaluation_task.construct()
        else:
            self.evaluation_task = None
        self.seed = seed
        self.project_name = _project_name
        self.entity = _entity
        self.save_path = _save_path
        self.wandb_logs = _wandb_logs
        self.exp_name = exp_name
        self.output_dir = os.path.join(_save_path, exp_name)
        self.optimizer = optimizer
        self.dataset = self.dataset.construct()
        self._config_copy = _config_copy

    def setup(self):
        # initial setup
        self.set_seed()
        os.makedirs(self.output_dir, exist_ok=True)

        # setup train args
        self.train_args = self.train_args.construct(data_seed=self.seed, seed=self.seed, output_dir=self.output_dir,
                                                    logging_dir=self.output_dir)
        # enable TF32
        if torch.cuda.is_available() and self.train_args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # setup dataset
        tokenized_datasets = self.dataset.load_tokenized_dataset()

        # setup trainer
        self._creat_optimizer(model=self.model)
        if self.wandb_logs:
            wandb_callback = [WandbCallback(model=self.model, entity=self.entity, project=self.project_name,
                                            name=self.exp_name, config=self._config_copy, tags=[])]
        else:
            wandb_callback = None

        # 5- setup logging
        self.setup_logging(log_path=self.output_dir, training_args=self.train_args)

        self.trainer = MyHFTrainer(model=self.model,
                                   args=self.train_args,
                                   optimizers=[self.optimizer, None],
                                   callbacks=wandb_callback,
                                   tokenizer=self.dataset.tokenizer,
                                   train_dataset=tokenized_datasets["train"],
                                   eval_dataset=tokenized_datasets["validation"],
                                   evaluation_task=self.evaluation_task,
                                   )

        jsonnet_string = json.dumps(self._config_copy, indent=4)
        save_path = os.path.join(self.output_dir, 'config.jsonnet')
        with open(save_path, 'w') as jsonnet_file:
            jsonnet_file.write(jsonnet_string)
        logger.info(f'configuration file saved at: {save_path}')

    def set_seed(self, seed=None):
        seed = seed if seed else self.seed
        # for reproducibility, we need to use deterministic torch cuda backend
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def setup_logging(log_path, training_args):

        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()

        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
        log_level = training_args.get_process_log_level()
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        file_formatter = transformers.utils.logging.logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", )
        file_handler = transformers.utils.logging.logging.FileHandler(os.path.join(log_path, f'logfile{os.environ.get("SLURM_JOBID", "")}.log'))
        file_handler.setFormatter(file_formatter)
        transformers.utils.logging.logging.root.addHandler(file_handler)

    def _checkpoint_is_available(self):
        items = os.listdir(self.trainer.args.output_dir)
        checkpoint_found = any(
            item.startswith("checkpoint") and os.path.isdir(os.path.join(self.trainer.args.output_dir, item)) for item
            in items)
        return checkpoint_found

    def train(self):

        self.setup()
        if self._checkpoint_is_available():
            logging.info(f'load from checkpoint and resume training')
            train_result = self.trainer.train(resume_from_checkpoint=True)
        else:
            train_result = self.trainer.train()
        print(train_result)
        self.trainer.save_state()
        state_dict = self.trainer.model.state_dict()
        if self.train_args.should_save:
            cpu_state_dict = {}
            for key in state_dict.keys():
                cpu_state_dict[key] = state_dict[key]
            del state_dict
            self.trainer._save(os.path.join(self.save_path, self.exp_name), state_dict=cpu_state_dict)  # noqa

    def load_and_tokenize_dataset(self):
        tokenized_datasets = self.dataset.create_tokenized_datasets()
        print(tokenized_datasets)

    @staticmethod
    def _auto_fill_args_with_hf_training_args(params: Params, train_args: MyTrainingArguments, special_values=None):
        from dataclasses import asdict
        key_dict = {'lr': 'learning_rate', 'weight_decay': 'weight_decay'}
        if special_values is None:
            special_values = ['auto']
        for k in params:
            if params.get(k) in special_values:
                params.__setitem__(k, asdict(train_args)[key_dict[k]])

    def _creat_optimizer(self, model):

        if len(self.optimizer._params) > 0:
            from transformers.trainer_pt_utils import get_parameter_names
            from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

            self._auto_fill_args_with_hf_training_args(self.optimizer._params, self.train_args)
            decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            quantizable_parameters = []
            for name, m in model.named_modules():
                if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.pytorch_utils.Conv1D):
                    if hasattr(m, 'weight'):
                        quantizable_parameters.append(f'{name}.weight')

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters() if
                        (n in quantizable_parameters) and (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.optimizer._params.as_dict()['weight_decay'],
                    "quantize": True,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if
                        (n not in quantizable_parameters) and (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.optimizer._params.as_dict()['weight_decay'],
                    "quantize": False,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "quantize": False,
                },
            ]

            self.optimizer = self.optimizer.construct(model_params=optimizer_grouped_parameters, quantizable_parameters=quantizable_parameters,
                                                      betas=[self.train_args.adam_beta1, self.train_args.adam_beta2],
                                                      eps=self.train_args.adam_epsilon,
                                                      )
        else:
            self.optimizer = None

    def evaluate(self, chk: int = None):

        from transformers.utils import WEIGHTS_NAME
        if chk is None:
            # load final checkpoint
            checkpoint_path = os.path.join(self.output_dir, WEIGHTS_NAME)
        else:
            checkpoint_path = os.path.join(self.output_dir, f"spec-checkpoint-{chk}", WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(checkpoint_path))
        print(f"load checkpoint from: {checkpoint_path}")

        accelerator = Accelerator(mixed_precision='bf16')
        self.model = accelerator.prepare_model(self.model)
        self.model.eval()

        result = self.evaluation_task.evaluate(docs=self.evaluation_task.validation_docs(),
                                               lm=self.model,
                                               tokenizer=self.dataset.tokenizer,
                                               )
        print(f'{self.evaluation_task.TASK_NAME}=====> \n{result}')
        return result
