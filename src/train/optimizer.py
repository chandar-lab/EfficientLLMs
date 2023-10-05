import torch
import torch.nn as nn
from common import FromParams, Registrable, Params, Lazy
from typing import Iterator


class Optimizer(Registrable):
    def __init__(self, lr: float, weight_decay: float,):
        self.lr = lr
        self.weight_decay = weight_decay

    def build(self, parameters):
        raise NotImplementedError()


@Optimizer.register('SGD')
class SGD(Optimizer):
    def __init__(self, momentum: float, nesterov: bool, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.momentum = momentum
        self.nesterov = nesterov

    def build(self, parameters):
        return torch.optim.SGD(parameters, self.lr, momentum=self.momentum,
                               weight_decay=self.weight_decay, nesterov=self.nesterov)


@Optimizer.register('Adam')
class Adam(Optimizer):
    def __init__(self, **kwargs):
        super(Adam, self).__init__(**kwargs)

    def build(self, parameters):
        return torch.optim.Adam(parameters, self.lr, weight_decay=self.weight_decay)


@Optimizer.register('RMSprop')
class RMSprop(Optimizer):
    def __init__(self, **kwargs):
        super(RMSprop, self).__init__(**kwargs)

    def build(self, parameters):
        return torch.optim.RMSprop(parameters, self.lr, weight_decay=self.weight_decay)

