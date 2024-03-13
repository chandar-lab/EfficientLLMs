import copy
import os
import sys
import random
import numpy as np
import torch
import wandb
import logging
import re
import csv

import json
from models import Base_Model, dequant_model
from data_loader import Dataset
from common import FromParams, Lazy, Params, load_configs
from typing import Dict, Any

from train import MyTrainingArguments
from transformers import Trainer
from train import MyHFTrainer
from callbacks import WandbCallback
import transformers
import datasets
from transformers.trainer_utils import get_last_checkpoint
from optimizer import Base_Optimizer
from evaluate import Evaluate
from plot import activation_hook, plot_mse_layer, extract_mse_between_layers, plot_eval_on_checkpoints

logger = logging.getLogger(__name__)


class Runtime(FromParams):
    def __init__(self, seed: int, _project_name: str, _entity: str, model: Lazy[Base_Model],
                 evaluation_task: Lazy[Evaluate],
                 train_args: Lazy[MyTrainingArguments], dataset: Lazy[Dataset], optimizer: Lazy[Base_Optimizer],
                 _save_path: str = './save', _wandb_logs: bool = False):
        self.model = model
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
        self.exp_name = None
        self.optimizer = optimizer
        os.makedirs(_save_path, exist_ok=True)

    def setup(self, EXPERIMENT_NAME: str, cfg: Dict[str, Any]):
        # 0- initial setup
        self.set_seed()
        self.exp_name = EXPERIMENT_NAME
        output_dir = os.path.join(self.save_path, EXPERIMENT_NAME)
        os.makedirs(output_dir, exist_ok=True)

        # 1- setup train args
        self.train_args = self.train_args.construct(data_seed=self.seed, seed=self.seed, output_dir=output_dir,
                                                    logging_dir=output_dir)
        # enable TF32
        if torch.cuda.is_available() and self.train_args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # 2- setup model
        model = self.model.construct(exp_name=self.exp_name, save_path=output_dir)

        # 3- setup dataset
        self.dataset = self.dataset.construct()
        tokenized_datasets = self.dataset.creat_tokenized_datasets()
        # tokenized_datasets = {'train': None, 'validation':None}

        # 4- setup trainer
        self._creat_optimizer(model=model)
        if self.wandb_logs:
            wandb_callback = [WandbCallback(model=model, entity=self.entity, project=self.project_name,
                                            name=EXPERIMENT_NAME, config=cfg, tags=[])]
        else:
            wandb_callback = None

        # 5- setup logging
        self.setup_logging(log_path=output_dir, training_args=self.train_args)

        self.trainer = MyHFTrainer(model=model,
                                   args=self.train_args,
                                   optimizers=[self.optimizer, None],
                                   callbacks=wandb_callback,
                                   tokenizer=self.dataset.tokenizer,
                                   train_dataset=tokenized_datasets["train"],
                                   eval_dataset=tokenized_datasets["validation"],
                                   evaluation_task=self.evaluation_task,
                                   )

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
            self.trainer._save(os.path.join(self.save_path, self.exp_name), state_dict=cpu_state_dict)  # noqa

    # Evaluation
    def evaluate(self, chk: int = None, exp_name: str = None):

        # load checkpoint
        if chk is None:
            # load last checkpoint
            checkpoint_path = get_last_checkpoint(self.trainer.args.output_dir)
        else:
            if exp_name is None:
                exp_name = self.exp_name
            checkpoint_path = os.path.join(self.save_path, exp_name, f"checkpoint-{chk}",
                                           "pytorch_model.bin")
        print(f"load checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, )
        self.trainer.model.load_state_dict(checkpoint)

        metrics = self.trainer.evaluate()
        max_eval_samples = len(self.trainer.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(self.trainer.eval_dataset))

        if self.evaluation_task is not None:
            metrics["eval_task"] = self.evaluation_task.compute(self.trainer.model, self.trainer.tokenizer)

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

    def plot_mse_between_layers(self):
        model = self.trainer.model
        model.eval()
        tokenized_datasets = self.dataset.creat_tokenized_datasets()
        input_ids = torch.tensor([tokenized_datasets['validation'][0]['input_ids']]).to('cuda')
        input_activation_qaunt, output_activation_qaunt = activation_hook(model, input_ids)

        model_dequant = copy.deepcopy(model)
        dequant_model(model_dequant)
        input_activation, output_activation = activation_hook(model_dequant, input_ids)
        c_attn, c_proj, mlp_c_fc, mlp_c_proj = extract_mse_between_layers(model_dequant, output_activation,
                                                                          output_activation_qaunt)
        plot_mse_layer(c_attn, c_proj, mlp_c_fc, mlp_c_proj, os.path.join(self.save_path, self.exp_name), self.exp_name)
        return (c_attn + c_proj + mlp_c_fc + mlp_c_proj).sum()

    def evaluation_on_checkpoints(self, eval_config: str = None):

        import pandas as pd

        all_files = os.listdir(os.path.join(self.save_path, self.exp_name))
        tmp_spec_checkpoints = [file for file in all_files if re.match(r'tmp-spec-checkpoint-\d+', file)]
        tmp_spec_checkpoints.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        global_iters = [int(re.search(r'\d+', file).group()) for file in tmp_spec_checkpoints]

        if eval_config is not None:
            t = load_configs(eval_config)['evaluation_task']
            task_name = t['type']
            self.trainer.evaluation_task = Evaluate.from_params(Params(t))
        else:
            task_name = None
        def _update_metrics(dict, new_dict):
            for key, value in new_dict.items():
                if key in dict:
                    dict[key].append(value)
                else:
                    dict[key] = [value]

        metrics = {}
        for i in range(len(tmp_spec_checkpoints)):
            checkpoint_path = os.path.join(self.save_path, self.exp_name, tmp_spec_checkpoints[i])
            print(f"load checkpoint from: {checkpoint_path}")
            self.trainer._load_from_checkpoint(checkpoint_path)
            """
            This flag should be set into True, to avoid changing the model into bf16. 
            Since, in this code, we used mixed precision, parameters are stored in fp32 and
            only specific operations modified into bf16 using torch-autocast. For the baseline model,
            we didn't face any issue. But for quantized model, quantization and dequantization process 
            make a huge difference. 
            """
            self.trainer.is_in_train = True
            res = self.trainer.evaluate()

            # modify keys
            if 'eval_runtime' in res.keys():
                res.pop('eval_runtime')
                res.pop('eval_samples_per_second')
                res.pop('eval_steps_per_second')
            new_dict = {}
            for key, value in res.items():
                if key.startswith('eval_ds'):
                    new_key = f'{task_name}_{key[len("eval_ds"):]}'
                    new_dict[new_key] = value
                else:
                    new_dict[key] = value

            # update metric dict
            _update_metrics(metrics, new_dict)
            _update_metrics(metrics, {'iters': global_iters[i]})

            # save results
            df = pd.DataFrame(metrics)
            df.to_csv(os.path.join(self.trainer.args.output_dir, 'eval_on_checkpoints.csv'), index=False)

        # plot the result
        plot_eval_on_checkpoints(metrics, save_path=self.trainer.args.output_dir)
