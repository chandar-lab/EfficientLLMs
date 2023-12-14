import torch
import torch.nn as nn
import torch.nn.functional as F
from quantization import Quantizer


class Quantized_Linear(nn.Linear):
    def __init__(self, weight_quantize_module: Quantizer, act_quantize_module: Quantizer,
                 in_features, out_features, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)
        self.weight_quantize_module = weight_quantize_module
        self.act_quantize_module = act_quantize_module
        # for name, p in weight_quantize_module.named_parameters():
        #     self.register_parameter(name='weight_' + name, param=p)

    def forward(self, input):
        if self.weight_quantize_module is None:
            weight_quant = self.weight
        else:
            weight_quant = self.weight_quantize_module(self.weight)
        if self.act_quantize_module is None:
            input_quant = input
        else:
            input_quant = self.act_quantize_module(input)
        return F.linear(input_quant, weight_quant, self.bias)


class Quantized_Conv2d(nn.Conv2d):
    def __init__(self, weight_quantize_module: Quantizer, act_quantize_module: Quantizer,
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(Quantized_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                               dilation=dilation, groups=groups, bias=bias)
        self.weight_quantize_module = weight_quantize_module
        self.act_quantize_module = act_quantize_module
        # for name, p in weight_quantize_module.named_parameters():
        #     self.register_parameter(name='weight_' + name, param=p)

    def forward(self, input):
        if self.weight_quantize_module is None:
            weight_quant = self.weight
        else:
            weight_quant = self.weight_quantize_module(self.weight)
        if self.act_quantize_module is None:
            input_quant = input
        else:
            input_quant = self.act_quantize_module(input)
        return F.conv2d(input_quant, weight_quant, self.bias, self.stride, self.padding, self.dilation, self.groups)

