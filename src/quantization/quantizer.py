import torch
import torch.nn as nn
from common import Registrable


def get_scale(input: torch.tensor, N_bits: int = 2):
    """
    extract optimal scale based on statistics of the input tensor.
    from: https://arxiv.org/pdf/1805.06085.pdf
    Args:
        input: input real tensor
        N_bits: bit precision
    Returns:
        scale: optimal scale
    """
    assert N_bits in [2, 4, 8]
    z_typical = {'2bit': [0.311, 0.678], '4bit': [0.077, 1.013], '8bit': [0.027, 1.114]}
    z = z_typical[f'{N_bits}bit']
    c1, c2 = 1 / z[0], z[1] / z[0]
    std = input.std()
    mean = input.abs().mean()
    q_scale = c1 * std - c2 * mean
    return q_scale.detach()


def round_pass(input: torch.tensor):
    """
    Args:
        input: input tensor

    Returns:
        rounded tensor with STE for backward
    """
    y = input.round()
    y_grad = input
    return (y - y_grad).detach() + y_grad


def grad_scale(input: torch.tensor, scale: float):
    """
        Args:
            input: input tensor
            scale: gradient scale for backward

        Returns:
            rounded tensor with STE for backward
        """
    y = input
    y_grad = input * scale
    return (y - y_grad).detach() + y_grad


class ExtractScaleAndBias(Registrable):
    def __init__(self, symmetric: bool = True, minimum_range: float = 1e-8):
        """
        Base class for extracting quantization parameters: step_size and zero_point.
        Uses the maximum absolute value for linear quantization.
        Args:
            symmetric: boolean flag to indicate symmetric or asymmetric quantization
            minimum_range: minimum range for clipping function
        """
        self.symmetric = symmetric
        self.minimum_range = minimum_range

    def __call__(self, input: torch.Tensor, Qn: int, Qp: int):
        raise NotImplementedError


@ExtractScaleAndBias.register("per-tensor")
class PerTensor(ExtractScaleAndBias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, input, Qn, Qp):
        if self.symmetric:
            scale = input.abs().max().detach().clamp_(min=self.minimum_range)
            step_size = scale / Qp
            zero_point = torch.tensor(0.)
        else:
            scale = (input.max() - input.min()).detach().clamp_(min=self.minimum_range)
            step_size = scale / (2 * Qp)
            zero_point = torch.round((input.min().detach() / step_size) - Qn)
        return step_size, zero_point

@ExtractScaleAndBias.register("per-row")
class PerRow(ExtractScaleAndBias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, input, Qn, Qp):
        assert len(input.shape) > 1, "row-wise quantization is not supported for 1d tensor"

        if self.symmetric:
            scale = input.abs().max(dim=-1, keepdim=True)[0].detach().clamp_(min=self.minimum_range)
            step_size = scale / Qp
            zero_point = torch.tensor(0.)
        else:
            scale = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).detach().clamp_(
                min=self.minimum_range)
            step_size = scale / (2 * Qp)
            zero_point = torch.round((input.min(dim=-1, keepdim=True)[0].detach() / step_size) - Qn)
        return step_size, zero_point

@ExtractScaleAndBias.register("per-column")
class PerColumn(ExtractScaleAndBias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, input, Qn, Qp):
        assert len(input.shape) > 1, "column-wise quantization is not supported for 1d tensor"
        # the input shape is B,T,Cin and the step size will have a shape of T,1
        if self.symmetric:
            scale = input.abs().amax(dim=(0, 2), keepdim=True)[0].detach().clamp_(min=self.minimum_range)
            step_size = scale / Qp
            zero_point = torch.tensor(0.)
        else:
            scale = (input.amax(dim=(0, 2), keepdim=True)[0] - input.amin(dim=(0, 2), keepdim=True)[
                0]).detach().clamp_(min=self.minimum_range)
            step_size = scale / (2 * Qp)
            zero_point = torch.round(
                (input.amin(dim=(0, 2), keepdim=True)[0].detach() / step_size) - Qn)
        return step_size, zero_point

@ExtractScaleAndBias.register("per-group")
class PerGroup(ExtractScaleAndBias):
    def __init__(self, groupsize: int = 32, **kwargs):
        super().__init__(**kwargs)
        assert groupsize > 1
        self.groupsize = groupsize

    def __call__(self, input, Qn, Qp):
        if self.groupsize > input.shape[-1]:
            self.groupsize = input.shape[-1]
        assert self.groupsize > 1
        assert input.shape[-1] % self.groupsize == 0
        assert input.dim() == 2
        to_quant = input.reshape(-1, self.groupsize)
        if self.symmetric:
            scales = to_quant.amax(dim=1, keepdim=True).detach().clamp_(min=self.minimum_range)
            step_size = (scales / Qp).reshape(input.shape[0], -1)
            zero_point = torch.tensor(0.)
        else:
            max_val = to_quant.amax(dim=1, keepdim=True).detach()
            min_val = to_quant.amin(dim=1, keepdim=True).detach()
            scales = (max_val - min_val).clamp(min=self.minimum_range) / (2 * Qp)
            zeros = torch.round((min_val / scales) - Qn)
            step_size = scales.reshape(input.shape[0], -1)
            zero_point = zeros.reshape(input.shape[0], -1)
        return step_size, zero_point


class Quantizer(Registrable):
    """
        Base class for quantizers, handling the bit width and signedness for quantization.

        Args:
            N_bits (int): Number of bits for quantization.
            signed (bool): Flag for signed (True) or unsigned (False) quantization.
        """
    def __init__(self, N_bits: int = 4, signed: bool = True):
        super().__init__()
        self.N_bits = N_bits
        self.signed = signed
        if self.signed:
            self.Qn = - 2 ** (self.N_bits - 1)
            self.Qp = 2 ** (self.N_bits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2 ** self.N_bits - 1

    def monitor_ranges(self):
        raise NotImplementedError

    def __call__(self, input: torch.Tensor):
        raise NotImplementedError


@Quantizer.register("linear")
class LinearQuantization(Quantizer):
    def __init__(self, get_scales: ExtractScaleAndBias, **kwargs):
        super().__init__(**kwargs)
        self.get_scales = get_scales
        self.activation_grad_flag = False

    def linear_quantize(self, input, step_size, zero_point):
        if type(self.get_scales) == PerGroup:
            to_quant = input.reshape(-1, self.groupsize)
            step_size = step_size.reshape(-1, 1)
            zero_point = zero_point.reshape(-1, 1)
            x = torch.clamp((to_quant / step_size) - zero_point, self.Qn, self.Qp).round()
            x = x.reshape_as(input)
        else:
            x = torch.clamp((input / step_size) - zero_point, self.Qn, self.Qp).round()
        return x

    def linear_dequantize(self, input, step_size, zero_point):
        if type(self.get_scales) == PerGroup:
            input_grouped = input.reshape(-1, self.groupsize)
            step_size = step_size.reshape(-1, 1)
            zero_point = zero_point.reshape(-1, 1)
            return ((input_grouped + zero_point) * step_size).reshape_as(input)
        else:
            return (input + zero_point) * step_size

    def __call__(self, input):
        if self.N_bits >= 32:
            return input
        else:
            step_size, zero_point = self.get_scales(input, self.Qn, self.Qp)
            quantized_input = self.linear_quantize(input, step_size, zero_point)
            dequantized_input = self.linear_dequantize(quantized_input, step_size, zero_point)
            return dequantized_input


@Quantizer.register("lsq")
class LSQ(Quantizer):
    def __init__(self, use_grad_scaled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_grad_scaled = use_grad_scaled
        self.step_size = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.zero_point = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def linear_quantize(self, input: torch.tensor):
        if self.use_grad_scaled:
            s_grad_scale = 1.0 / ((self.Qp * input.numel()) ** 0.5)
            step_size = grad_scale(self.step_size, s_grad_scale)
        else:
            step_size = self.step_size
        x = torch.clamp((input / step_size) - self.zero_point, self.Qn, self.Qp)
        x = round_pass(x)
        return x

    def linear_dequantize(self, input: torch.tensor):
        return (input + self.zero_point) * self.step_size

    def _init_q_params(self, input: torch.tensor):
        self.step_size.data = input.detach().abs().mean() * 2 / ((2 ** (self.N_bits - 1) - 1) ** 0.5)

    def monitor_ranges(self):
        return {'max_weight': self.max_range, 'min_weight': self.min_range,
                'range_pos': (self.step_size * self.Qp).item(), 'range_neg': (self.step_size * self.Qn).item()}

    def __call__(self, input: torch.Tensor):
        if self.N_bits >= 32:
            return input
        else:
            quantized_input = self.linear_quantize(input)
            dequantized_input = self.linear_dequantize(quantized_input)
            return dequantized_input

@Quantizer.register("split_quant")
class Split_Quantization(LinearQuantization):
    """
        Quantization class that extends LinearQuantization to handle split quantization.

        Args:
            num_split (int): Number of splits for the input tensor columns.

        Note:
            In FlashAttention models, the qkv projections are often combined, leading to
            different gradient ranges. This class addresses the issue by separating the qkv
            projections into chunks and quantizing them individually, which enhances the
            gradient quantization performance.
        """
    def __init__(self, num_split: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_split = num_split

    @staticmethod
    def split_tensor_columns(input, num_split):
        cols_per_split = input.size(1) // num_split
        split_tensors = []
        for i in range(num_split):
            start_index = i * cols_per_split
            end_index = (i + 1) * cols_per_split if i < num_split - 1 else input.size(1)
            split_tensor = input[:, start_index:end_index]
            split_tensors.append(split_tensor)
        return split_tensors

    def forward(self, input: torch.tensor):
        assert input.ndim == 2
        if self.N_bits >= 32:
            return input
        else:
            if self.num_split > 1:
                splits = self.split_tensor_columns(input, self.num_split)
                quantized_splits = [self.linear_dequantize(self.linear_quantize(_)) for _ in splits]

                return torch.cat(quantized_splits, dim=1)
            else:
                quantized_input = self.linear_quantize(input)
                dequantized_input = self.linear_dequantize(quantized_input)
                return dequantized_input


class _quantize_global(torch.autograd.Function):
    """
        Custom autograd function for quantizing inputs, weights, and gradients for
        global quantization in neural networks. This function supports forward and
        backward passes with optional quantization modules for weights, activations,
        and gradients.

        Methods:
            forward(ctx, X_3D, W, bias, w_qmodule, a_qmodule, g_qmodule):
                Applies the forward pass with optional quantization for weights and activations.

            backward(ctx, grad_output):
                Computes the backward pass with optional quantization for gradients.

        Notes:
            - The input tensor is reshaped for matrix multiplication and then reshaped back to the original
              dimensions after the operation.
            - The `activation_grad_flag` in the gradient quantization module (`g_qmodule`) determines whether
              to apply quantization to the gradient with respect to the input or weights.
            - This code is modified from the bitsandbytes repository:
              https://github.com/TimDettmers/bitsandbytes/blob/73d3e7b61307a7a8c05a8bab1be7a54d4ebd0156/bitsandbytes/nn/triton_based_modules.py#L69.
        """

    @staticmethod
    def forward(ctx, X_3D, W, bias=None, w_qmodule=None, a_qmodule=None, g_qmodule=None):
        # reshape input to [N * L, D]
        X = X_3D.view(-1, X_3D.size(-1))
        if w_qmodule is not None:
            weight_quant = w_qmodule(W)
        else:
            weight_quant = W
        if a_qmodule is not None:
            input_quant = a_qmodule(X)
        else:
            input_quant = X

        # save for backward.
        ctx.save_for_backward = X, W

        ctx.g_qmodule = g_qmodule
        output = input_quant.matmul(weight_quant.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output.view(*X_3D.size()[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):

        G = grad_output.reshape(-1, grad_output.size(-1))
        grad_X = grad_W = grad_bias = None
        X, W = ctx.save_for_backward

        if ctx.g_qmodule is not None:
            grad_quant = ctx.g_qmodule(G)
        else:
            grad_quant = G

        if ctx.needs_input_grad[0]:
            if ctx.g_qmodule is not None and ctx.g_qmodule.activation_grad_flag:
                grad_X = torch.matmul(grad_quant, W.to(grad_quant.dtype)).view(*grad_output.size()[:-1], -1)
            else:
                grad_X = torch.matmul(G, W.to(G.dtype)).view(*grad_output.size()[:-1], -1)
        if ctx.needs_input_grad[1]:
            if ctx.g_qmodule is not None and ctx.g_qmodule.activation_grad_flag:
                grad_W = torch.matmul(G.t(), X.to(G.dtype))
            else:
                grad_W = torch.matmul(grad_quant.t(), X.to(grad_quant.dtype))
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        return grad_X, grad_W, grad_bias, None, None, None


