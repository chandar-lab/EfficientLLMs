import torch
import argparse
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

try:
    from flash_attn.modules.mha import MHA
except ImportError:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention as MHA

try:
    from flash_attn.models.gpt import GPTLMHeadModel
except ImportError:
    from transformers.models.gpt2 import GPT2LMHeadModel as GPTLMHeadModel

from pytorch_memory_profiler import forward_benchmark, forward_backward_benchmark, full_benchmark, activation_hook, \
    GB_scale, forward_backward_benchmark_data, forward_benchmark_data


def args_parser():
    parser = argparse.ArgumentParser(
        description='profile time',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--seq_len', default=1024, type=int, help='sequence length')
    parser.add_argument('--repeats', default=10, type=int, help='number of repeat in profiling')
    parser.add_argument('--model', default='gpt2-small', type=str, help='name of model')
    parser.add_argument('--save_path', default='../profile_results', type=str, help='save path')
    parser.add_argument('--amp', dest='amp', action='store_true', help='use AMP')
    parser.add_argument('--forward', action='store_true', help='profile for forward only')
    parser.add_argument('--forward_backward', action='store_true', help='profile for forward and backward')
    parser.add_argument('--categorizes', dest='categorizes', action='store_true',
                        help='use categorizes profiling to indication param, activation, grad, state, ...')
    parser.add_argument('--compile', action='store_true', help='use torch.compile and TF cores')
    parser.add_argument('--verbose', action='store_true', help='show profile time and memory results')
    parser.add_argument('--hf', action='store_true', help='Use HuggingFace model')
    parser.add_argument('--cpu', action='store_true', help='force to use cpu device')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = args_parser()
    os.makedirs(args.save_path, exist_ok=True)
    if torch.cuda.is_available() and not args.cpu:
        device_type = 'cuda'
    else:
        if not args.hf:
            raise "FlashAttention model only works on GPU"
        device_type = 'cpu'

    if args.model == 'gpt2-small':
        config = GPT2Config()
    elif args.model == 'gpt2-medium':
        config = GPT2Config(n_embd=1024, n_head=16, n_layer=24)
    elif args.model == 'gpt2-large':
        config = GPT2Config(n_embd=1280, n_head=20, n_layer=36)
    else:
        raise 'not provide'
    if args.hf:
        # atten = GPT2Attention(config).cuda()
        model = GPT2LMHeadModel(config)
        atten = model.transformer.h[0].cuda()
        prefix_peak_memory_name = os.path.join(args.save_path, f"{args.model.replace('gpt2', 'MHA')}_HF_{args.seq_len}")
    else:
        config.use_flash_attn = True
        config.fused_bias_fc = False
        config.fused_mlp = False
        config.fused_dropout_add_ln = False
        config.residual_in_fp32 = True
        config.pad_vocab_size_multiple = 8
        prefix_peak_memory_name = os.path.join(args.save_path, f"{args.model.replace('gpt2', 'MHA')}_{args.seq_len}")
        # atten = MHA(embed_dim=config.n_embd,
        #                  num_heads=12,
        #                  num_heads_kv=12,
        #                  dropout=0.1,
        #                  causal=True,
        #                  layer_idx=0,
        #                  use_flash_attn=True,
        #                  softmax_scale=0.125,
        #                 ).cuda()
        model = GPTLMHeadModel(config)
        # atten = model.transformer.layers[0].cuda()
        atten = model.transformer.embeddings.cuda()

    if args.compile:
        torch.set_float32_matmul_precision('high')
        atten = torch.compile(atten)

    # estimated memory:
    # batch = torch.randn(args.batch_size, args.seq_len, config.n_embd)
    batch = torch.randint(0,100, (args.batch_size, args.seq_len))
    output_activation = activation_hook(atten, batch.to(device_type), device_type=device_type,
                                        amp=args.amp)
    activation_size = sum([v.numel() * v.element_size() for k, v in output_activation.items() if
                           k.startswith('transformer') or k.startswith('lm_head')]) / GB_scale
    logit_size = args.batch_size * args.seq_len * config.vocab_size * 4 / GB_scale
    parameter_size = sum([p.numel() * p.element_size() for p in atten.parameters()]) / GB_scale
    states_size = sum([p.numel() * p.element_size() * 2 for p in atten.parameters() if p.requires_grad]) / GB_scale
    print(
        f"========= estimated memory usage =========\n"
        f"PARAMETER:: {parameter_size},\n"
        f"OPTIMIZER_STATE:: {states_size},\n"
        f"ACTIVATION:: {activation_size},\n"
        f"LOGIT_SIZE: {logit_size},\n"
        f"TOTAL: {activation_size + parameter_size + states_size + logit_size}\n"
        f"==========================================="
    )

    if args.forward:
        # data = torch.randn(args.batch_size, args.seq_len, config.n_embd)
        data = torch.randint(0,100, (args.batch_size, args.seq_len))
        forward_benchmark_data(atten, data, repeats=args.repeats, amp=args.amp, device_type=device_type,
                               categorizes=args.categorizes, prefix_peak_memory_name=prefix_peak_memory_name,
                               verbose=args.verbose)
    elif args.forward_backward:
        # data = torch.randn(args.batch_size, args.seq_len, config.n_embd)
        data = torch.randint(0,100, (args.batch_size, args.seq_len))
        forward_backward_benchmark_data(atten, data, repeats=args.repeats, amp=args.amp, device_type=device_type,
                                        categorizes=args.categorizes, prefix_peak_memory_name=prefix_peak_memory_name,
                                        verbose=args.verbose)
    else:
        raise NotImplementedError
