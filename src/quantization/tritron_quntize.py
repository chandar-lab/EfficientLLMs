# code form BitsAndBytes library: https://github.com/TimDettmers/bitsandbytes/tree/main/bitsandbytes/triton

from common import FromParams, Registrable, Params
import triton
import triton.language as tl
from triton.language.math import llrint
import math
import torch
from quantization import Quantizer


# global quantize
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, }, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048, }, num_stages=1),

    ],
    key=['n_elements']
)
@triton.jit
def _quantize_global(
        x_ptr,
        absmax_inv_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)
    output = tl.libdevice.llrint(127. * (x * absmax_inv))
    tl.store(output_ptr + offsets, output, mask=mask)


def quantize_global(x: torch.Tensor):
    absmax = x.abs().max().unsqueeze(0)
    absmax_inv = 1. / absmax
    output = torch.empty(*x.shape, device='cuda', dtype=torch.int8)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _quantize_global[grid](x, absmax_inv, output, n_elements)
    return output, absmax


# global quantize and transpose
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=4),

        # ...
    ],
    key=['M', 'N']
)
@triton.jit
def _quantize_global_transpose(A, absmax_inv_ptr, B, stride_am, stride_an, stride_bn, stride_bm, M, N,
                               BLOCK_M: tl.constexpr,
                               BLOCK_N: tl.constexpr,
                               GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)

    # rematerialize to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    B = B + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    output = llrint(127. * (a * absmax_inv))

    tl.store(B, output, mask=mask)


def quantize_global_transpose(input):
    absmax = input.abs().max().unsqueeze(0)
    absmax_inv = 1. / absmax
    M, N = input.shape
    out = torch.empty(N, M, device='cuda', dtype=torch.int8)

    assert out.size(0) == N and out.size(1) == M
    assert input.stride(0) == 1 or input.stride(1) == 1
    assert out.stride(0) == 1 or out.stride(1) == 1

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _quantize_global_transpose[grid](input, absmax_inv, out, input.stride(0), input.stride(1), out.stride(0),
                                     out.stride(1), M, N)
    return out, absmax


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=8, num_warps=8),
        triton.Config({}, num_stages=1),
        triton.Config({}, num_stages=2),
        triton.Config({}, num_stages=4),
        triton.Config({}, num_stages=8),
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['n_elements']
)
@triton.jit
def _quantize_rowwise(
        x_ptr,
        output_ptr,
        output_maxs,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)

    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    # output = tl.libdevice.llrint(127. * (x / max_val))
    output = llrint(127. * (x / max_val))
    tl.store(output_ptr + offsets, output, mask=row_mask)
    tl.store(output_maxs + pid, max_val)


def quantize_rowwise(x: torch.Tensor):
    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
    output_maxs = torch.empty(x.shape[0], device=x.device, dtype=torch.float16)

    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (x.shape[0],)
    _quantize_rowwise[grid](x, output, output_maxs, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output, output_maxs


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=8, num_warps=8),
        triton.Config({}, num_stages=1),
        triton.Config({}, num_stages=2),
        triton.Config({}, num_stages=4),
        triton.Config({}, num_stages=8),
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['n_elements']
)
@triton.jit
def _dequantize_rowwise(
        x_ptr,
        state_x,
        output_ptr,
        inv_127,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)
    max_val = tl.load(state_x + pid)
    output = max_val * x * inv_127
    tl.store(output_ptr + offsets, output, mask=row_mask)


def dequantize_rowwise(x: torch.Tensor, state_x: torch.Tensor):
    output = torch.empty(*x.shape, device=x.device, dtype=torch.float16)

    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (x.shape[0],)
    _dequantize_rowwise[grid](x, state_x, output, 1. / 127, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1),
        triton.Config({}, num_stages=2),
        triton.Config({}, num_stages=4),
        triton.Config({}, num_stages=8),
        triton.Config({}, num_stages=16),
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=8, num_warps=8),
        triton.Config({}, num_stages=16, num_warps=8),
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['n_elements']
)
@triton.jit
def _quantize_columnwise_and_transpose(
        x_ptr,
        output_ptr,
        output_maxs,
        n_elements,
        M: tl.constexpr, N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid
    p2_arange = tl.arange(0, P2)
    p2_arange_mask = p2_arange < M
    arange = p2_arange * N
    offsets = block_start + arange
    x = tl.load(x_ptr + offsets, mask=p2_arange_mask)
    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(p2_arange_mask, abs_x, 0), axis=0)
    output = llrint(127. * (x / max_val))

    new_start = pid * M
    new_offsets = new_start + p2_arange
    tl.store(output_ptr + new_offsets, output, mask=p2_arange_mask)
    tl.store(output_maxs + pid, max_val)


def quantize_columnwise_and_transpose(x: torch.Tensor):
    M, N = x.shape
    output = torch.empty(N, M, device=x.device, dtype=torch.int8)
    output_maxs = torch.empty(x.shape[1], device=x.device, dtype=torch.float16)

    P2 = int(2 ** (math.ceil(math.log2(M))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _quantize_columnwise_and_transpose[grid](x, output, output_maxs, n_elements, M, N, BLOCK_SIZE=M, P2=P2)
    return output, output_maxs


@Quantizer.register("triton_8bit")
class Triton_8bit(Quantizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = None
        self.size = None
        assert self.N_bits == 8

    def linear_quantize(self, input: torch.tensor):
        if self.granularity == 'per-tensor':
            output, self.state = quantize_rowwise(input)
            return output

        elif self.granularity == 'per-column':
            assert len(input.shape) > 1
            if len(input.shape) == 3:
                X = input.view(-1, input.size(-1))
                self.size = input.size()
                output, self.state = quantize_rowwise(X)
                del X
            else:
                output, self.state = quantize_rowwise(input)
                self.size = None
            return output
        else:
            raise NotImplementedError

    def linear_dequantize(self, input: torch.tensor):
        output = dequantize_rowwise(input, self.state)
        if self.size is not None:
            return output.view(*self.size[:-1], -1)
        else:
            return output
