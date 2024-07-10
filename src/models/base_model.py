import torch
import torch.nn as nn
import wandb
from quantization import Quantizer, Quantized_Linear
from common import FromParams, Registrable, Params, Lazy
import logging
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


def make_quant_layer(weight_quantize_module: Quantizer, act_quantize_module: Quantizer, grad_quantize_module: Quantizer,
                     module: torch.nn.Module, layer_type: torch.nn):
    """
    code modified from: https://github.com/IST-DASLab/gptq
    recursively replace the layers of the model with `layer_type`, with quantized version one
    Args:
        weight_quantize_module: quantizer module for weight
        act_quantize_module: quantizer module for activation
        grad_quantize_module: quantizer module for gradient
        module: model
        layer_type: nn.Linear or nn.Conv2d or transformers.pytorch_utils.Conv1D

    Returns:

    """
    if isinstance(module, Quantized_Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) == layer_type:
            setattr(
                module, attr,
                construct_quant_layer(weight_quantize_module, act_quantize_module, grad_quantize_module, tmp, layer_type)
            )
            quant_layer = module.__getattr__(attr)
            if tmp.bias is not None:
                quant_layer.bias.data = tmp.bias.data

    for name1, child in module.named_children():
        make_quant_layer(weight_quantize_module, act_quantize_module, grad_quantize_module, child, layer_type)


def construct_quant_layer(weight_quantize_module: Quantizer, act_quantize_module: Quantizer, grad_quantize_module: Quantizer,
                          layer: torch.nn.Module, layer_type: torch.nn):
    if weight_quantize_module is None:
        wq_module = None
    else:
        wq_module = weight_quantize_module.construct()
    if act_quantize_module is None:
        act_module = None
    else:
        act_module = act_quantize_module.construct()
    if grad_quantize_module is None:
        gard_module = None
    else:
        gard_module = grad_quantize_module.construct()

    if layer_type == nn.Linear:
        quant_layer = Quantized_Linear(wq_module, act_module, gard_module, layer.in_features, layer.out_features,
                                       layer.bias is not None)
        quant_layer.weight.data = layer.weight.data
        return quant_layer

    elif layer_type == Conv1D:
        quant_layer = Quantized_Linear(wq_module, act_module, gard_module, layer.weight.shape[0], layer.weight.shape[1],
                                       layer.bias is not None)
        quant_layer.weight.data = layer.weight.data.T
        return quant_layer
    else:
        raise NotImplementedError


class Base_Model(Registrable):
    def __init__(self, weight_quantize_module: Lazy[Quantizer], act_quantize_module: Lazy[Quantizer],
                 grad_quantize_module: Lazy[Quantizer]):
        # super().__init__()
        if len(weight_quantize_module._params) > 0:
            self.weight_quantize_module = weight_quantize_module
        else:
            self.weight_quantize_module = None
        if len(act_quantize_module._params) > 0:
            self.act_quantize_module = act_quantize_module
        else:
            self.act_quantize_module = None
        if len(grad_quantize_module._params) > 0:
            self.grad_quantize_module = grad_quantize_module
        else:
            self.grad_quantize_module = None
        if self.act_quantize_module is not None or self.weight_quantize_module is not None or self.grad_quantize_module is not None:
            self._model2quant()

    def _model2quant(self):
        make_quant_layer(self.weight_quantize_module, self.act_quantize_module, self.grad_quantize_module, self, nn.Linear)
        make_quant_layer(self.weight_quantize_module, self.act_quantize_module, self.grad_quantize_module, self, Conv1D)

        # split the qkv grad quantization
        for name, m in self.named_modules():
            if 'Wqkv' in name and hasattr(m, "grad_quantize_module") and m.grad_quantize_module is not None:
                m.grad_quantize_module.num_split = 3

    def loglikelihood(self, context, continuation, tokenizer, truncate=True):
        # when too long to fit in context, truncate from the left
        inp = torch.tensor([tokenizer.encode(context + continuation)[-1024:]], dtype=torch.long).to(self.device)
        ctxlen = len(tokenizer.encode(context.strip()))
        ctxlen = min(ctxlen, 1023)

        cont_toks = inp[:, ctxlen:]  # [batch, seq]
        logits = F.log_softmax(self.forward(inp)[0], dim=-1)[:, ctxlen - 1:-1]  # [batch, seq, vocab]

        return torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1).detach().cpu().sum()

    def monitoring_range(self, wandb_logs=False):
        items = {}
        for name, m in self.named_modules():
            if isinstance(m, Quantized_Linear):
                res = m.weight_quantize_module.monitor_ranges()
                for key, value in res.items():
                    items[f'{name}_{key}'] = value
        if wandb_logs:
            wandb.log(items)
        logging.info(items)
