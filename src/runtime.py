import os
import random
import numpy as np
import torch
import wandb
import logging

import json
from models import Base_Model
from data_loader import Dataset
from common import FromParams, Lazy
from typing import Dict, Any
import math

from train import MyTrainingArguments
from transformers import Trainer


class Runtime(FromParams):
    def __init__(self, seed: int, _project_name: str, _entity: str, model: Lazy[Base_Model],
                 train_args: Lazy[MyTrainingArguments], dataset: Lazy[Dataset], _save_path: str = './save',
                 _wandb_logs: bool = False, _resume: str = None, _device: str = 'cuda'):
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
        os.makedirs(_save_path, exist_ok=True)

    def setup(self, EXPERIMENT_NAME: str, cfg: Dict[str, Any]):
        self.set_seed()
        self.exp_name = EXPERIMENT_NAME
        output_dir = os.path.join(self.save_path, EXPERIMENT_NAME)
        os.makedirs(output_dir, exist_ok=True)
        if self.wandb_logs:
            os.environ["WANDB_PROJECT"] = self.project_name  # log to your project
            os.environ["WANDB_LOG_MODEL"] = "all"  # log your models
            self.train_args = self.train_args.construct(data_seed=self.seed, seed=self.seed, output_dir=output_dir,
                                                        logging_dir=output_dir, run_name=EXPERIMENT_NAME, report_to=['wandb'])
        else:
            self.train_args = self.train_args.construct(data_seed=self.seed, seed=self.seed, output_dir=output_dir,
                                                        logging_dir=output_dir, run_name=EXPERIMENT_NAME,
                                                        report_to=None)

        model = self.model.construct(exp_name=self.exp_name, save_path=output_dir)
        if not torch.cuda.is_available() and 'cuda' in self.device:
            logging.warning('CUDA not available')
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.device)
        self.dataset = self.dataset.construct()
        tokenized_datasets = self.dataset.creat_tokenized_datasets()
        data_collator = self.dataset.creat_data_collator()
        self.trainer = Trainer(model=model,
                               tokenizer=self.dataset.tokenizer,
                               args=self.train_args,
                               train_dataset=tokenized_datasets["train"],
                               eval_dataset=tokenized_datasets["validation"],
                               data_collator=data_collator)

        self.setup_logging(log_path=os.path.join(self.save_path, EXPERIMENT_NAME))

        jsonnet_string = json.dumps(cfg, indent=4)
        save_path = os.path.join(self.save_path, self.exp_name, 'config.jsonnet')
        with open(save_path, 'w') as jsonnet_file:
            jsonnet_file.write(jsonnet_string)
        logging.info(f'configuration file saved at: {save_path}')

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
    def setup_logging(log_path):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(log_path, 'logfile.log'))
        file_handler.setLevel(logging.INFO)  # Set the desired logging level
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger('').addHandler(file_handler)

    def train(self):
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

    # def load_model(self):
    #     if self.resume is not None:
    #         # load checkpoint to resume training
    #         model_path = os.path.join(self.save_path, self.exp_name, f'{self.resume}')
    #         logging.info(f"=> loading checkpoint from: {model_path}")
    #         if os.path.exists(model_path):
    #             checkpoint = torch.load(model_path, map_location=self.model.device)
    #             self.model.load_state_dict(checkpoint['model_state_dict'])
    #             self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #             self.trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #             resume_epoch = checkpoint['epoch']
    #             return resume_epoch
    #         else:
    #             logging.warning(f"model_path: {model_path} didn't exist")
    #             raise "path to model didn't exist."
    #     else:
    #         model_path = os.path.join(self.save_path, self.exp_name, 'model_checkpoint.pth')
    #         logging.info(f"=> loading checkpoint from: {model_path}")
    #         if os.path.exists(model_path):
    #             loaded_state_dict = torch.load(model_path, map_location=self.model.device)
    #             network_kvpair = self.model.state_dict()
    #             for key in loaded_state_dict.keys():
    #                 network_kvpair[key] = loaded_state_dict[key]
    #             self.model.load_state_dict(network_kvpair)
    #         else:
    #             logging.warning(f"model_path: {model_path} didn't exist")
    #             raise "path to model didn't exist."
    #
    # def evaluate(self):
    #     train_loader, test_loader, valid_loader, calibration_loader = self.dataset.build()
    #     test_loss, test_acc = evaluate(self.model, test_loader)
    #     if self.wandb_logs:
    #         wandb.log({"test_accuracy": test_acc, "test_loss": test_loss})
    #     logging.info(f"test accuracy: {test_acc}%, test loss: {test_loss}")
    #
    # def load_and_evaluate(self):
    #     self.load_model()
    #     train_loader, test_loader, valid_loader, calibration_loader = self.dataset.build()
    #     test_loss, test_acc = evaluate(self.model, test_loader)
    #     if self.wandb_logs:
    #         wandb.log({"test_accuracy": test_acc, "test_loss": test_loss})
    #     logging.info(f"test accuracy: {test_acc}%, test loss: {test_loss}")
    #
    # def resume_and_train(self):
    #     self.trainer.build(self.model)
    #     resume_epoch = self.load_model()
    #     if hasattr(self.dataset, 'use_train'):
    #         self.dataset.use_train = True
    #     train_loader, test_loader, valid_loader, calibration_loader = self.dataset.build()
    #     self.trainer.fit(self.model, train_loader, valid_loader, resume_epoch=resume_epoch)
