import torch
import numpy as np
import wandb
import logging
import os
from collections import deque

from .optimizer import Optimizer
from .scheduler import Scheduler
from common import FromParams, Registrable, Params, Lazy
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def evaluate(model, data_loader, criterion=torch.nn.CrossEntropyLoss()):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for (images, labels) in data_loader:
            output = model.modified_forward(images.to(model.device))
            loss = criterion(output, labels.to(model.device))
            prec1, prec5 = accuracy(output.data, labels.to(model.device), topk=(1, 5))
            losses.update(loss.detach().item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

    return losses.avg, top1.avg

class Trainer(FromParams):
    def __init__(self, optimizer: Optimizer, scheduler: Scheduler,
                 num_epochs: int,
                 wandb_logs: bool, max_mag_coef: float,
                 _monitoring_range: bool, grad_clip: float = None,
                 q_range_weight_decay: float = None, save_path: str = './save',
                 _max_checkpoints: int = 5, exp_name: str = 'exp'):

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.wandb_logs = wandb_logs
        self.max_mag_coef = max_mag_coef
        self.monitoring_range = _monitoring_range
        self.grad_clip = grad_clip
        self.q_range_weight_decay = q_range_weight_decay
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_path = save_path
        self.exp_name = exp_name
        self.best_valid_accuracy = 0.
        self.max_checkpoints = _max_checkpoints
        self.saved_checkpoints = deque(maxlen=_max_checkpoints+1)

    def _set_q_range_weight_decay(self, model: torch.nn.Module):
        model_params = []
        for name, p in model.named_parameters():
            if "weight_q_range" in name:
                model_params.append({'params': p, 'weight_decay': self.q_range_weight_decay})
            else:
                model_params.append({'params': p})
        return model_params

    def build(self, model: torch.nn.Module):
        model_params = self._set_q_range_weight_decay(model)
        self.optimizer = self.optimizer.build(model_params)
        self.scheduler = self.scheduler.build(self.optimizer)

    def save_checkpoint(self, model, epoch, val_accuracy):
        checkpoint_dir = os.path.join(self.save_path, self.exp_name)
        os.makedirs(checkpoint_dir, exist_ok=True)  # Create the directory if it doesn't exist

        checkpoint_filename = f"model_checkpoint_epoch_{epoch}.tar"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_valid_accuracy': val_accuracy,
        }

        torch.save(checkpoint, checkpoint_path)
        self.saved_checkpoints.append(checkpoint_path)

        # Remove old checkpoints if we exceed the maximum number
        if len(self.saved_checkpoints) >= self.max_checkpoints:
            old_checkpoint_path = self.saved_checkpoints.popleft()
            os.remove(old_checkpoint_path)

    def save_best_model_checkpoint(self):
        # Get a list of all saved checkpoints
        checkpoint_dir = os.path.join(self.save_path, self.exp_name)
        checkpoint_files = os.listdir(checkpoint_dir)
        checkpoint_files = [f for f in checkpoint_files if f.startswith("model_checkpoint_epoch_")]

        if not checkpoint_files:
            return

        # Sort checkpoints by epoch number
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        # Rename the last checkpoint to "model_checkpoint.pth"
        last_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
        new_checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pth")

        checkpoint = torch.load(last_checkpoint_path)
        model_state_dict = checkpoint['model_state_dict']
        torch.save(model_state_dict, new_checkpoint_path)

    def train_one_epoch(self, model, train_loader):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for (images, labels) in train_loader:

            output = model.modified_forward(images.to(model.device))  # TODO: on_blocking=True ?
            loss = self.criterion(output, labels.to(model.device))
            prec1, prec5 = accuracy(output.data, labels.to(model.device), topk=(1, 5))
            losses.update(loss.detach().item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

            if self.max_mag_coef is not None:
                loss += self.max_mag_coef * model.compute_max_magnitude_loss()
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip)
            self.optimizer.step()
        self.scheduler.step()

        return losses.avg, top1.avg

    def fit(self, model, train_loader, valid_loader, resume_epoch=0):
        with logging_redirect_tqdm():
            for epoch in trange(resume_epoch, self.num_epochs):
                train_loss, train_acc = self.train_one_epoch(model, train_loader)
                valid_loss, valid_acc = evaluate(model, valid_loader, self.criterion)
                if self.wandb_logs:
                    wandb.log({"train_loss": train_loss, "train_accuracy": train_acc,
                               "valid_loss": valid_loss, "valid_accuracy": valid_acc})
                logging.info(f"Epoch {epoch + 1}/{self.num_epochs} - "
                             f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
                             f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.2f}%")
                if valid_acc > self.best_valid_accuracy:
                    self.best_valid_accuracy = valid_acc
                    self.save_checkpoint(model, epoch, valid_acc)
                if self.monitoring_range:
                    model.monitoring_range(self.wandb_logs)

        self.save_best_model_checkpoint()


