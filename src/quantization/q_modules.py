import torch.nn as nn
from quantization import Quantizer, _quantize_global

class Quantized_Linear(nn.Linear):
    def __init__(self, weight_quantize_module: Quantizer, act_quantize_module: Quantizer, grad_quantize_module: Quantizer,
                 in_features, out_features, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)
        self.weight_quantize_module = weight_quantize_module
        self.act_quantize_module = act_quantize_module
        self.grad_quantize_module = grad_quantize_module

    def forward(self, input):
        return _quantize_global.apply(input, self.weight, self.bias, self.weight_quantize_module,
                                      self.act_quantize_module, self.grad_quantize_module)
