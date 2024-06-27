import torch
import torch.nn as nn
import gc
import pandas as pd
import torch
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import CrossEntropyLoss
from flash_attn.models.gpt import GPTLMHeadModel
from transformers import GPT2Config, GPT2LMHeadModel
import argparse
import os
from flash_attn.modules.mha import MHA


def args_parser():
    parser = argparse.ArgumentParser(
        description='profile time',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--seq_len', default=1024, type=int, help='sequence length')
    parser.add_argument('--repeats', default=10, type=int, help='number of repeat in profiling')
    parser.add_argument('--model', default='gpt2-small', type=str, help='name of model')
    parser.add_argument('--save_path', default='../save', type=str, help='save path')
    parser.add_argument('--amp', dest='amp', action='store_true', help='use AMP')
    parser.add_argument('--bf16', action='store_true', help='use bf16')
    parser.add_argument('--hf', action='store_true', help='use hf model')
    parser.add_argument('--forward', dest='forward', action='store_true', help='profile for foward only')
    args = parser.parse_args()
    return args


def full_model_forward_backward(x, labels, model):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        lm_logits = model(x).logits
        labels = labels.to(lm_logits.device)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss


def forward_benchmark(
        fn, *inputs, repeats=10, desc="", verbose=False, amp=True, amp_dtype=torch.bfloat16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(repeats):
        amp_wrapper(*inputs, **kwinputs)
    torch.cuda.cudart().cudaProfilerStop()


def forward_backward_benchmark(
        fn,
        *inputs,
        grad=None,
        repeats=10,
        desc="",
        verbose=False,
        amp=True,
        amp_dtype=torch.bfloat16,
        **kwinputs,
):
    """Use Pytorch Benchmark on the backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Backward pass")

    def amp_wrapper(grad, *inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            y = fn(*inputs, **kwinputs)
            if type(y) is tuple:
                y = y[0]
            if grad is None:
                grad = torch.randn_like(y)
            else:
                if grad.shape != y.shape:
                    raise RuntimeError("Grad shape does not match output shape")
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
            y.backward(grad, retain_graph=True)

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(repeats):
        amp_wrapper(grad, *inputs, **kwinputs)
    torch.cuda.cudart().cudaProfilerStop()


# Should run with:
# nsys profile -w true -t cuda -s cpu --capture-range=cudaProfilerApi --capture-range-end stop -x true -f true -o linear_fb_amp python nsight_profile.py --amp
# Then for creating report:
# nsys stats --output report_linear_fb_amp --report gpukernsum --force-overwrite true linear_fb_amp.nsys-rep
# report can be gpumemtimesum, gpumemsizesum, gpukernsum
if __name__ == "__main__":
    assert torch.cuda.is_available()
    # toy example
    # batch_size = 8
    # num_training_steps = 3
    # device = torch.device('cuda')
    # model = nn.Sequential(
    #     nn.Linear(768, 2304),
    #     nn.ReLU(),
    #     nn.Linear(2304, 768),
    # )
    # # criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss()
    # optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    # model = model.to(device)
    # model.train()
    # torch.cuda.cudart().cudaProfilerStart()
    # for step in range(num_training_steps):
    #     x = torch.randn(batch_size, 1024, 768).to(device)
    #     label = torch.randn(batch_size, 1024, 768).to(device)
    #     output = model(x)
    #     loss = criterion(output, label)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     scheduler.step()
    # torch.cuda.cudart().cudaProfilerStop()

    # # ALL GPT2 forward and backward
    # args = args_parser()
    # if args.model == 'gpt2-small':
    #     config = GPT2Config()
    # elif args.model == 'gpt2-medium':
    #     config = GPT2Config(n_embd=1024, n_head=16, n_layer=24)
    # elif args.model == 'gpt2-large':
    #     config = GPT2Config(n_embd=1280, n_head=20, n_layer=36)
    # else:
    #     raise 'not provide'
    # config.use_flash_attn = True
    # config.fused_bias_fc = True
    # config.fused_mlp = True
    # config.fused_dropout_add_ln = True
    # config.residual_in_fp32 = True
    # config.pad_vocab_size_multiple = 8
    # model = GPTLMHeadModel(config).cuda()

    # torch.cuda.cudart().cudaProfilerStart()
    # for step in range(args.repeats):
    #     x = torch.randint(0, 50257, (args.batch_size, args.seq_len)).to('cuda')
    #     label = torch.randint(0, 50257, (args.batch_size, args.seq_len)).to('cuda')
    #     loss = full_model_forward_backward(x, label, model)
    #     loss.backward()
    # torch.cuda.cudart().cudaProfilerStop()

    # # Multi head attention forward and backward
    # args = args_parser()
    # mha_module = MHA(embed_dim=768,
    #              num_heads=12,
    #              num_heads_kv=12,
    #              dropout=0.1,
    #              causal=True,
    #              layer_idx=0,
    #              use_flash_attn=True,
    #              softmax_scale=0.125,
    #             ).cuda()
    # x = torch.randn(args.batch_size, args.seq_len, 768).to('cuda').requires_grad_(True).to(torch.bfloat16)
    # if args.bf16:
    #     mha_module = mha_module.to(torch.bfloat16)
    # torch.cuda.cudart().cudaProfilerStart()
    # forward_backward_benchmark(mha_module, x, repeats=args.repeats, amp=args.amp)
    # torch.cuda.cudart().cudaProfilerStop()

    # Attention Block
    from flash_attn.models.gpt import create_block

    args = args_parser()
    if args.model == 'gpt2-small':
        config = GPT2Config()
    elif args.model == 'gpt2-medium':
        config = GPT2Config(n_embd=1024, n_head=16, n_layer=24)
    elif args.model == 'gpt2-large':
        config = GPT2Config(n_embd=1280, n_head=20, n_layer=36)
    elif args.model == 'gpt2-xlarge':
        config = GPT2Config(n_embd=1600, n_head=25, n_layer=48)
    else:
        raise 'not provide'
    if args.hf:
        # atten = GPT2Attention(config).cuda()
        model = GPT2LMHeadModel(config)
        attention_block = model.transformer.h[0].cuda()
    else:
        config.use_flash_attn = True
        config.fused_bias_fc = True
        config.fused_mlp = True
        config.fused_dropout_add_ln = True
        config.residual_in_fp32 = True
        config.pad_vocab_size_multiple = 8
        attention_block = create_block(config, layer_idx=0, process_group=None).cuda()

    x = torch.randn(args.batch_size, args.seq_len, config.n_embd).to(device='cuda',
                                                                     dtype=torch.bfloat16).requires_grad_(True)
    if args.bf16:
        attention_block = attention_block.to(torch.bfloat16)
    torch.cuda.cudart().cudaProfilerStart()
    forward_backward_benchmark(attention_block, x, repeats=args.repeats, amp=args.amp)
    torch.cuda.cudart().cudaProfilerStop()

    # # Linear layer
    # args = args_parser()
    # m_linear = torch.nn.Linear(768, 2304).cuda()
    # x = torch.randn(args.batch_size, args.seq_len, 768).to('cuda').requires_grad_(True)
    # if args.amp or args.bf16:
    #     x = x.to(torch.bfloat16)
    # if args.bf16:
    #     m_linear = m_linear.to(torch.bfloat16)
    # if args.forward:
    #     forward_benchmark(m_linear, x, repeats=args.repeats, amp=args.amp)
    # else:
    #     forward_backward_benchmark(m_linear, x, repeats=args.repeats, amp=args.amp)
