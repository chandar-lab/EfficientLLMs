import os
import sys
import random
import numpy as np
import torch
import wandb
import logging

import json
from models import Base_Model
from data_loader import Dataset
from common import FromParams, Lazy, Params
from typing import Dict, Any
import math

from train import MyTrainingArguments
from transformers import Trainer
import transformers
import datasets
from transformers.integrations.integration_utils import WandbCallback
from transformers import TrainerCallback
from optimizer import Base_Optimizer

logger = logging.getLogger(__name__)


class Runtime(FromParams):
    def __init__(self, seed: int, _project_name: str, _entity: str, model: Lazy[Base_Model],
                 train_args: Lazy[MyTrainingArguments], dataset: Lazy[Dataset], optimizer: Lazy[Base_Optimizer],
                 _save_path: str = './save', _wandb_logs: bool = False, _resume: str = None, _device: str = 'cuda'):
        self.model = model
        self.train_args = train_args
        self.trainer = None
        self.dataset = dataset
        self.seed = seed
        self.project_name = _project_name
        self.entity = _entity
        self.save_path = _save_path
        self.wandb_logs = _wandb_logs
        self.exp_name = None
        self.resume = _resume
        self.device = _device
        self.optimizer = optimizer
        os.makedirs(_save_path, exist_ok=True)

    def setup(self, EXPERIMENT_NAME: str, cfg: Dict[str, Any]):
        # 0- initial setup
        self.set_seed()
        self.exp_name = EXPERIMENT_NAME
        output_dir = os.path.join(self.save_path, EXPERIMENT_NAME)
        os.makedirs(output_dir, exist_ok=True)

        # 1- setup train args
        if self.wandb_logs:
            os.environ["WANDB_PROJECT"] = self.project_name  # log to your project
            os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
            self.train_args = self.train_args.construct(data_seed=self.seed, seed=self.seed, output_dir=output_dir,
                                                        logging_dir=output_dir, run_name=EXPERIMENT_NAME,
                                                        report_to=['wandb'])
        else:
            os.environ["WANDB_DISABLED"] = "true"
            self.train_args = self.train_args.construct(data_seed=self.seed, seed=self.seed, output_dir=output_dir,
                                                        logging_dir=output_dir, run_name=EXPERIMENT_NAME,
                                                        report_to=None)

        # 2- setup model
        model = self.model.construct(exp_name=self.exp_name, save_path=output_dir)
        if self.train_args.bf16:
            logger.info("change model to bf16.")
            model = model.to(dtype=torch.bfloat16, device=self.train_args.device)
        if not torch.cuda.is_available() and 'cuda' in self.device:
            logging.warning('CUDA not available')
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.device)

        # 3- setup dataset
        self.dataset = self.dataset.construct()
        tokenized_datasets = self.dataset.creat_tokenized_datasets()
        # tokenized_datasets = {'train': None, 'validation':None}

        # 4- setup trainer
        self._creat_optimizer(model=model)
        self.trainer = Trainer(model=model,
                               args=self.train_args,
                               optimizers=[self.optimizer, None],
                               train_dataset=tokenized_datasets["train"],
                               eval_dataset=tokenized_datasets["validation"])

        # 5- setup logging and wandb and callbacks
        self.setup_logging(log_path=os.path.join(self.save_path, EXPERIMENT_NAME), training_args=self.train_args)
        if self.wandb_logs:
            # init wandb callback and update the config
            for callback in self.trainer.callback_handler.callbacks:
                if isinstance(callback, WandbCallback):
                    callback.setup(self.trainer.args, self.trainer.state, self.trainer.model)
                    if self.trainer.state.is_world_process_zero:
                        callback._wandb.config.update(cfg, allow_val_change=True)

        jsonnet_string = json.dumps(cfg, indent=4)
        save_path = os.path.join(self.save_path, self.exp_name, 'config.jsonnet')
        with open(save_path, 'w') as jsonnet_file:
            jsonnet_file.write(jsonnet_string)
        logger.info(f'configuration file saved at: {save_path}')

    def set_seed(self):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def setup_logging(log_path, training_args):

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()

        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        file_handler = logging.FileHandler(os.path.join(log_path, 'logfile.log'))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%m/%d/%Y %H:%M:%S"))
        # logger.addHandler(file_handler)
        logging.root.addHandler(file_handler)

    def _checkpoint_is_available(self):
        items = os.listdir(self.trainer.args.output_dir)
        checkpoint_found = any(
            item.startswith("checkpoint") and os.path.isdir(os.path.join(self.trainer.args.output_dir, item)) for item
            in items)
        return checkpoint_found

    def train(self):
        if self._checkpoint_is_available():
            logging.info(f'load from checkpoint and resume training')
            train_result = self.trainer.train(resume_from_checkpoint=True)
        else:
            train_result = self.trainer.train()
        self.trainer.save_state()
        state_dict = self.trainer.model.state_dict()
        if self.train_args.should_save:
            cpu_state_dict = {}
            for key in state_dict.keys():
                cpu_state_dict[key] = state_dict[key]
            del state_dict
            trainer._save(os.path.join(self.save_path, EXPERIMENT_NAME), state_dict=cpu_state_dict)  # noqa

    # Evaluation
    def evaluate(self):
        metrics = self.trainer.evaluate()
        max_eval_samples = len(self.trainer.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(self.trainer.eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

    def check_dataset(self):
        tokenized_datasets = self.dataset.creat_tokenized_datasets()
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
        from transformers.utils.import_utils import is_sagemaker_mp_enabled
        # if is_sagemaker_mp_enabled:
        #     raise NotImplementedError

        if len(self.optimizer._params) > 0:
            from transformers.trainer_pt_utils import get_parameter_names
            from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

            self._auto_fill_args_with_hf_training_args(self.optimizer._params, self.train_args)
            decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.optimizer._params.as_dict()['weight_decay'],
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = self.optimizer.construct(model_params=optimizer_grouped_parameters,
                                                      betas=[self.train_args.adam_beta1, self.train_args.adam_beta2],
                                                      eps=self.train_args.adam_epsilon,
                                                      )
        else:
            self.optimizer = None
