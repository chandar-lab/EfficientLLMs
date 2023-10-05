import torch
import torch.nn as nn
from common import FromParams, Registrable, Params, Lazy
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer


class Scheduler(Registrable):
    def __init__(self):
        pass

    def build(self, optimizer: Optimizer):
        raise NotImplementedError()

    def step(self):
        pass


@Scheduler.register('CosineAnnealingLR')
class CosineAnnealingLR(Scheduler):
    def __init__(self, T_max: int, eta_min: float):
        super(CosineAnnealingLR, self).__init__()
        self.T_max = T_max
        self.eta_min = eta_min

    def build(self, optimizer: Optimizer):
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.eta_min)


@Scheduler.register('ExponentialLR')
class ExponentialLR(Scheduler):
    def __init__(self, gamma: float):
        super(ExponentialLR, self).__init__()
        self.gamma = gamma

    def build(self, optimizer: Optimizer):
        return lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)


@Scheduler.register('MultiStepLR')
class MultiStepLR(Scheduler):
    def __init__(self, gamma: float, milestones: list):
        super(MultiStepLR, self).__init__()
        self.gamma = gamma
        self.milestones = milestones

    def build(self, optimizer: Optimizer):
        return lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)


# TODO: null scheduler
