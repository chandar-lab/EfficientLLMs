import torch
import torch.nn as nn
import wandb
from quantization import Quantizer, Quantized_Conv2d, Quantized_Linear
from common import FromParams, Registrable, Params, Lazy
import os
import logging
from typing import Iterator
from transformers.pytorch_utils import Conv1D


def make_quant_layer(weight_quantize_module: Quantizer, act_quantize_module: Quantizer,
                     module: torch.nn.Module, layer_type: torch.nn):
    """
    code modified from: https://github.com/IST-DASLab/gptq
    recursively replace the layers of the model with `layer_type`, with quantized version one
    Args:
        weight_quantize_module: quantizer module for weight
        act_quantize_module: quantizer module for activation
        module: model
        layer_type: nn.Linear or nn.Conv2d or transformers.pytorch_utils.Conv1D

    Returns:

    """
    if isinstance(module, Quantized_Conv2d) or isinstance(module, Quantized_Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) == layer_type:
            setattr(
                module, attr,
                construct_quant_layer(weight_quantize_module, act_quantize_module, tmp, layer_type)
            )
            quant_layer = module.__getattr__(attr)
            if tmp.bias is not None:
                quant_layer.bias.data = tmp.bias.data

    for name1, child in module.named_children():
        make_quant_layer(weight_quantize_module, act_quantize_module, child, layer_type)


def construct_quant_layer(weight_quantize_module: Quantizer, act_quantize_module: Quantizer,
                          layer: torch.nn.Module, layer_type: torch.nn):
    if weight_quantize_module is None:
        wq_module = None
    else:
        wq_module = weight_quantize_module.construct()
        wq_module._init_q_params(layer.weight)
    if act_quantize_module is None:
        act_module = None
    else:
        act_module = act_quantize_module.construct()

    if layer_type == nn.Linear:
        quant_layer = Quantized_Linear(wq_module, act_module, layer.in_features, layer.out_features,
                                       layer.bias is not None)
        quant_layer.weight.data = layer.weight.data
        return quant_layer

    elif layer_type == nn.Conv2d:
        quant_layer = Quantized_Conv2d(wq_module, act_module, layer.in_channels, layer.out_channels, layer.kernel_size,
                                       stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                                       groups=layer.groups, bias=layer.bias is not None)
        quant_layer.weight.data = layer.weight.data
        return quant_layer

    elif layer_type == Conv1D:
        quant_layer = Quantized_Linear(wq_module, act_module, layer.weight.shape[0], layer.weight.shape[1],
                                       layer.bias is not None)
        quant_layer.weight.data = layer.weight.data.T
        return quant_layer
    else:
        raise NotImplementedError


def dequant_model(model):
    for name, m in model.named_modules():
        if isinstance(m, Quantized_Linear):
            if m.weight_quantize_module is not None:
                m.weight_quantize_module.N_bits = 32
            if m.act_quantize_module is not None:
                m.act_quantize_module.N_bits = 32

class Base_Model(Registrable):
    def __init__(self, weight_quantize_module: Lazy[Quantizer], act_quantize_module: Lazy[Quantizer], exp_name: str = None,
                 save_path: str = './save'):
        # super().__init__()
        self.exp_name = exp_name
        self.save_path = save_path
        if len(weight_quantize_module._params) > 0:
            self.weight_quantize_module = weight_quantize_module
        else:
            self.weight_quantize_module = None
        if len(act_quantize_module._params) > 0:
            self.act_quantize_module = act_quantize_module
        else:
            self.act_quantize_module = None
        if self.act_quantize_module is not None or self.weight_quantize_module is not None:
            self._model2quant()

    def _model2quant(self):
        make_quant_layer(self.weight_quantize_module, self.act_quantize_module, self, nn.Linear)
        make_quant_layer(self.weight_quantize_module, self.act_quantize_module, self, Conv1D)

    def compute_max_magnitude_loss(self):
        loss = torch.tensor(0.).to(self.device)
        for name, m in self.named_modules():
            if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
                if m.bias is not None:
                    loss += (m.weight.abs().max() + m.bias.abs().max())
                else:
                    loss += m.weight.abs().max()
        return loss

    def monitoring_range(self, wandb_logs=False):
        items = {}
        for name, m in self.named_modules():
            if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
                res = m.weight_quantize_module.monitor_ranges()
                for key, value in res.items():
                    items[f'{name}_{key}'] = value
        if wandb_logs:
            wandb.log(items)
        logging.info(items)
