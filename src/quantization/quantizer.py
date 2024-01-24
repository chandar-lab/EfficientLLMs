import torch
import torch.nn as nn
from common import FromParams, Registrable, Params
import numpy as np
from scipy.optimize import minimize
import json


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


class _Clamp(torch.autograd.Function, Registrable):

    @staticmethod
    def forward(ctx, input, q_range, signed=True):
        """
        Args:
            ctx: a context object that can be used to stash information for backward computation
            input: input tensor
            signed: flag to indicate signed ot unsigned quantization
            q_range: scale factor

        Returns:
            clipped tensor
        """
        ctx.q_range = q_range
        ctx.input = input.clone()
        if signed:
            return input.clamp(-q_range, q_range)
        else:
            return input.clamp(torch.tensor(0.).to(q_range.device), q_range)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


@_Clamp.register("ste")
class _Clamp_STE(_Clamp):

    @staticmethod
    def backward(ctx, grad_output):
        q_range_grad = -1. * (ctx.input < -ctx.q_range) + 1. * (ctx.input > ctx.q_range)
        input_grad = 1.
        return input_grad * grad_output, q_range_grad * grad_output, None


@_Clamp.register("pwl")
class _Clamp_PWL(_Clamp):

    @staticmethod
    def backward(ctx, grad_output):
        q_range_grad = -1 * (ctx.input < -ctx.q_range) + 1 * (ctx.input > ctx.q_range)
        input_grad = 1. * (ctx.input.abs() <= ctx.q_range) + 0. * (ctx.input.abs() > ctx.q_range)
        return input_grad * grad_output, q_range_grad * grad_output, None


@_Clamp.register("mad")
class _Clamp_MAD(_Clamp):

    @staticmethod
    def backward(ctx, grad_output):
        q_range_grad = -1 * (ctx.input < -ctx.q_range) + 1 * (ctx.input > ctx.q_range)
        input_grad = 1. * (ctx.input.abs() <= ctx.q_range) + ctx.q_range / ctx.input.abs() * (
                ctx.input.abs() > ctx.q_range)
        return input_grad * grad_output, q_range_grad * grad_output, None


class Quantizer(torch.nn.Module, Registrable):
    def __init__(self, N_bits: int = 4, signed: bool = True, granularity: str = 'per-tensor', inplace: bool = False,
                 all_positive: bool = False, symmetric: bool = True, minimum_range: float = 1e-5):
        super().__init__()
        self.N_bits = N_bits
        self.signed = signed
        self.max_range = 0.
        self.min_range = 0.
        assert granularity in ['per-tensor', 'per-column', 'per-token', 'per-group']
        self.granularity = granularity
        self.inplace = inplace
        self.symmetric = symmetric
        self.minimum_range = minimum_range
        if all_positive:
            self.Qn = 0
            self.Qp = 2 ** self.N_bits - 1
        else:
            if self.signed:
                self.Qn = - 2 ** (self.N_bits - 1)
                self.Qp = 2 ** (self.N_bits - 1) - 1
            else:
                self.Qn = 0
                self.Qp = 2 ** self.N_bits - 1

    def linear_quantize(self, input: torch.tensor):
        raise NotImplementedError

    def linear_dequantize(self, input: torch.tensor):
        raise NotImplementedError

    def _init_q_params(self, input: torch.tensor):
        raise NotImplementedError

    def monitor_ranges(self):
        raise NotImplementedError

    def update_step_size(self, input):
        raise NotImplementedError

    def inplace_quantize(self, input):
        raise NotImplementedError

    def forward(self, input: torch.tensor):
        if self.N_bits >= 32:
            return input
        else:
            if self.inplace:
                return self.inplace_quantize(input)
            else:
                # for monitoring weights
                self.max_range = input.max().item()
                self.min_range = input.min().item()
                quantized_input = self.linear_quantize(input)
                dequantized_input = self.linear_dequantize(quantized_input)
                return dequantized_input


@Quantizer.register("normal")
class Normal(Quantizer):
    def __init__(self, beta: float = None, groupsize: int = None, **kwargs):
        super().__init__(**kwargs)
        self.step_size = torch.tensor(1.)
        self.zero_point = torch.tensor(0.)
        # TODO: to save and load step size and zero point, it's better to keep them as a register buffer.
        #  But DDP failed due to: "RuntimeError: No backend type associated with device type cpu".
        # self.register_buffer('step_size', torch.tensor(1.))
        # self.register_buffer('zero_point', torch.tensor(0.))
        # instead of using dynamic range, use this beta to keep track of range and use moving average of observed data to update step size
        self.beta = beta
        self.groupsize = groupsize

    def linear_quantize(self, input: torch.tensor):

        self.update_step_size(input)
        if self.granularity == 'per-group':
            to_quant = input.reshape(-1, self.groupsize)
            self.step_size = self.step_size.reshape(-1, 1)
            self.zero_point = self.zero_point.reshape(-1, 1)
            x = torch.clamp((to_quant / self.step_size) - self.zero_point, self.Qn, self.Qp)
            x = round_pass(x).reshape_as(input)
        else:
            x = torch.clamp((input / self.step_size) - self.zero_point, self.Qn, self.Qp)
            x = round_pass(x)
        return x

    def linear_dequantize(self, input: torch.tensor):
        if self.granularity == 'per-group':
            input_grouped = input.reshape(-1, self.groupsize)
            self.step_size = self.step_size.reshape(-1, 1)
            self.zero_point = self.zero_point.reshape(-1, 1)
            return ((input_grouped + self.zero_point) * self.step_size).reshape_as(input)
        else:
            return (input + self.zero_point) * self.step_size

    def _init_q_params(self, input: torch.tensor):
        self.update_step_size(input)

    def monitor_ranges(self):
        return {'max_weight': self.max_range, 'min_weight': self.min_range}

    @staticmethod
    def _step_size_moving_average_update(step_size_new, step_size_pre, beta):
        if beta is None:
            return step_size_new
        else:
            return step_size_pre * beta + step_size_new * (1 - beta)

    def update_step_size(self, input):
        if self.granularity == 'per-tensor':
            if self.symmetric:
                scale = input.abs().max().detach().clamp_(min=self.minimum_range)
                # self.step_size = scale / self.Qp
                self.step_size = self._step_size_moving_average_update(scale / self.Qp, self.step_size, self.beta)
                self.zero_point = torch.tensor(0.)
            else:
                scale = (input.max() - input.min()).detach().clamp_(min=self.minimum_range)
                # self.step_size = scale / (2*self.Qp)
                self.step_size = self._step_size_moving_average_update(scale / (2 * self.Qp), self.step_size, self.beta)
                self.zero_point = torch.round((input.min().detach() / self.step_size) - self.Qn)

        elif self.granularity == 'per-column':
            assert len(input.shape) > 1, "per channel quantization is not supported for 1d tensor"
            if len(input.shape) == 4:
                raise "It's not clear how to implement per channel for 4d tensors"
            # for 3d tensor, this implementation is not batch independent!

            if self.symmetric:
                scale = input.abs().max(dim=-1, keepdim=True)[0].detach().clamp_(min=self.minimum_range)
                # self.step_size = scale / self.Qp
                self.step_size = self._step_size_moving_average_update(scale / self.Qp, self.step_size, self.beta)
                self.zero_point = torch.tensor(0.)
            else:
                scale = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).detach().clamp_(
                    min=self.minimum_range)
                # self.step_size = scale / (2*self.Qp)
                self.step_size = self._step_size_moving_average_update(scale / (2 * self.Qp), self.step_size, self.beta)
                self.zero_point = torch.round((input.min(dim=-1, keepdim=True)[0].detach() / self.step_size) - self.Qn)

        elif self.granularity == 'per-token':
            assert len(input.shape) == 3, "per token quantization is only supported for 3d tensor"
            # the input shape is B,T,Cin and the step size will have a shape of T,1
            if self.symmetric:
                scale = input.abs().amax(dim=(0, 2), keepdim=True)[0].detach().clamp_(min=self.minimum_range)
                self.step_size = scale / self.Qp
                self.zero_point = torch.tensor(0.)
            else:
                scale = (input.amax(dim=(0, 2), keepdim=True)[0] - input.amin(dim=(0, 2), keepdim=True)[
                    0]).detach().clamp_(min=self.minimum_range)
                self.step_size = scale / (2 * self.Qp)
                self.zero_point = torch.round(
                    (input.amin(dim=(0, 2), keepdim=True)[0].detach() / self.step_size) - self.Qn)

        elif self.granularity == 'per-group':
            if self.groupsize > input.shape[-1]:
                self.groupsize = input.shape[-1]
            assert self.groupsize > 1
            assert input.shape[-1] % self.groupsize == 0
            assert input.dim() == 2
            to_quant = input.reshape(-1, self.groupsize)
            if self.symmetric:
                scales = to_quant.amax(dim=1, keepdim=True).detach().clamp_(min=self.minimum_range)
                self.step_size = (scales / self.Qp).reshape(input.shape[0], -1)
                self.zero_point = torch.tensor(0.)
            else:
                max_val = to_quant.amax(dim=1, keepdim=True).detach()
                min_val = to_quant.amin(dim=1, keepdim=True).detach()
                scales = (max_val - min_val).clamp(min=self.minimum_range) / (2 * self.Qp)
                zeros = torch.round((min_val / scales) - self.Qn)
                self.step_size = scales.reshape(input.shape[0], -1)
                self.zero_point = zeros.reshape(input.shape[0], -1)

        else:
            raise NotImplementedError

    def inplace_quantize(self, input):
        # for inference
        self.update_step_size(input)
        with torch.no_grad():
            input.div_(self.step_size).sub_(self.zero_point).round_().add_(self.zero_point).mul_(self.step_size)


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


@Quantizer.register("wcat")
class WCAT(Quantizer):
    def __init__(self, clip: _Clamp, use_grad_scaled: bool = True, init_method: str = 'max_abs',
                 noisy_mse_ber: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip
        self.use_grad_scaled = use_grad_scaled
        self.init_method = init_method
        self.noisy_mse_ber = noisy_mse_ber
        self.q_range = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.step_size = torch.tensor(1.)
        self.zero_point = torch.tensor(0.)

    def linear_quantize(self, input: torch.tensor):
        if self.use_grad_scaled:
            q_range_grad_scale = 1.0 / (input.numel() ** 0.5)
            q_range = grad_scale(self.q_range, q_range_grad_scale)
        else:
            q_range = self.q_range
        x = self.clip.apply(input, q_range, self.signed)
        if self.signed:
            self.step_size = q_range.detach() / (2 ** (self.N_bits - 1))
        else:
            self.step_size = q_range.detach() / (2 ** self.N_bits - 1)
        x_int = round_pass((x / self.step_size) - self.zero_point)
        x_clip = torch.clamp(x_int, self.Qn, self.Qp)
        return (x_clip - x_int).detach() + x_int

    def linear_dequantize(self, input: torch.tensor):
        return (input + self.zero_point) * self.step_size

    def _init_q_params(self, input: torch.tensor):
        if self.init_method == 'max_abs':
            self.q_range.data = input.detach().abs().max()
        elif self.init_method == 'SAWB':
            self.q_range.data = get_scale(input, self.N_bits)
        elif self.init_method == 'MSE':
            self.q_range.data = self._bruteforce_optimal_MSE(input)
        else:
            raise NotImplementedError

    def _bruteforce_optimal_MSE(self, input: torch.tensor):

        def mse(q_range_star: float = 1.):
            self.q_range.data = torch.tensor(q_range_star)
            input_quant = self.forward(input)
            return torch.nn.functional.mse_loss(input, input_quant).item()

        # q_range_ = np.array(max(input.abs().max().detach(), 1e-10))
        q_range_ = np.array(input.detach().abs().mean() * 2 * ((2 ** (self.N_bits - 1) - 1) ** 0.5))
        res = minimize(mse, q_range_, method='Nelder-Mead', tol=1e-6)
        assert res.success
        return torch.tensor(res.x[0])

    def monitor_ranges(self):
        return {'max_weight': self.max_range, 'min_weight': self.min_range,
                'range_pos': self.q_range.item(), 'range_neg': (-self.q_range).item()}


class ExtractScaleAndBias(Registrable):
    def __init__(self, symmetric: bool = True, Qp: int = None, Qn: int = None,
                 beta: float = None, minimum_range: float = 1e-8):
        """

        Args:
            symmetric: boolean flag to indicate symmetric or asymmetric quantization
            Qp: maximum quantize range
            Qn: minimum quantize range
            beta: to save the moving average of the scale
            minimum_range: minimum range for clipping function
        """
        self.symmetric = symmetric
        self.Qp = Qp
        self.Qn = Qn
        self.beta = beta
        self.minimum_range = minimum_range

    def return_scale_and_bias(self, input: torch.tensor, step_size):
        raise NotImplementedError

    def _step_size_moving_average_update(self, step_size_new, step_size_pre):
        if self.beta is None:
            return step_size_new
        else:
            return step_size_pre * self.beta + step_size_new * (1 - self.beta)


@ExtractScaleAndBias.register("per-tensor")
class PerTensor(ExtractScaleAndBias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def return_scale_and_bias(self, input: torch.tensor, step_size):
        if self.symmetric:
            scale = input.abs().max().detach().clamp_(min=self.minimum_range)
            step_size = self._step_size_moving_average_update(scale / self.Qp, step_size)
            zero_point = torch.tensor(0.)
        else:
            scale = (input.max() - input.min()).detach().clamp_(min=self.minimum_range)
            step_size = self._step_size_moving_average_update(scale / (2 * self.Qp), step_size)
            zero_point = torch.round((input.min().detach() / step_size) - self.Qn)
        return step_size, zero_point

@ExtractScaleAndBias.register("per-column")
class PerColumn(ExtractScaleAndBias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def return_scale_and_bias(self, input: torch.tensor, step_size):
        assert len(input.shape) > 1, "per channel quantization is not supported for 1d tensor"
        if len(input.shape) == 4:
            raise "It's not clear how to implement per channel for 4d tensors"
        # for 3d tensor, this implementation is not batch independent!

        if self.symmetric:
            scale = input.abs().max(dim=-1, keepdim=True)[0].detach().clamp_(min=self.minimum_range)
            step_size = self._step_size_moving_average_update(scale / self.Qp, step_size)
            zero_point = torch.tensor(0.)
        else:
            scale = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).detach().clamp_(
                min=self.minimum_range)
            step_size = self._step_size_moving_average_update(scale / (2 * self.Qp), step_size)
            zero_point = torch.round((input.min(dim=-1, keepdim=True)[0].detach() / step_size) - self.Qn)
        return step_size, zero_point

@ExtractScaleAndBias.register("per-token")
class PerToken(ExtractScaleAndBias):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def return_scale_and_bias(self, input: torch.tensor, step_size):
        assert len(input.shape) == 3, "per token quantization is only supported for 3d tensor"
        # the input shape is B,T,Cin and the step size will have a shape of T,1
        if self.symmetric:
            scale = input.abs().amax(dim=(0, 2), keepdim=True)[0].detach().clamp_(min=self.minimum_range)
            step_size = scale / self.Qp
            zero_point = torch.tensor(0.)
        else:
            scale = (input.amax(dim=(0, 2), keepdim=True)[0] - input.amin(dim=(0, 2), keepdim=True)[
                0]).detach().clamp_(min=self.minimum_range)
            step_size = scale / (2 * self.Qp)
            zero_point = torch.round(
                (input.amin(dim=(0, 2), keepdim=True)[0].detach() / step_size) - self.Qn)
        return step_size, zero_point

@ExtractScaleAndBias.register("per-group")
class PerGroup(ExtractScaleAndBias):
    def __init__(self, group_size: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size

    def return_scale_and_bias(self, input: torch.tensor, step_size):
        raise NotImplementedError
