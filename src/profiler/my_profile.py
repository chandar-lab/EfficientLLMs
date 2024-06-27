import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pandas as pd
import os
import gc
import argparse
from einops import rearrange, repeat
from torch.nn import CrossEntropyLoss

from flash_attn.models.gpt import GPTLMHeadModel
from transformers import GPT2Config

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_with_kvcache
from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined


def clean_up():
    torch.cuda.empty_cache()  # Empty the CUDA cache
    gc.collect()  # Collect garbage to free memory


def args_parser():
    parser = argparse.ArgumentParser(
        description='profile time',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--seq_len', default=1024, type=int, help='sequence length')
    parser.add_argument('--repeats', default=10, type=int, help='number of repeat in profiling')
    parser.add_argument('--model', default='gpt2-small', type=str, help='name of model')
    parser.add_argument('--save_path', default='../save', type=str, help='save path')
    args = parser.parse_args()
    return args


def full_model_forward_backward(x, labels, model):
    # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    lm_logits = model(x).logits
    labels = labels.to(lm_logits.device)
    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def model_forward_backward(x, model):
    lm_logits = model(x).logits
    return lm_logits


def attention_layer_forward_backward(x, model):
    out = model.transformer.layers[0](x)
    return out


def attention_layers_forward_backward(hidden_states, model):
    if model.transformer.parallel_block:
        hidden_states2 = None
    residual = None
    inference_params = None
    mixer_kwargs = (
        {"seqlen": hidden_states.shape[1]}
        if model.transformer.process_group is not None and model.transformer.sequence_parallel
        else {}
    )
    if inference_params is not None:
        mixer_kwargs["inference_params"] = inference_params
    for layer in model.transformer.layers:
        if model.transformer.prenorm:
            if not model.transformer.parallel_block:
                hidden_states, residual = layer(
                    hidden_states, residual, mixer_kwargs=mixer_kwargs
                )
            else:
                hidden_states, hidden_states2, residual = layer(
                    hidden_states, hidden_states2, residual, mixer_kwargs=mixer_kwargs
                )
        else:
            hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
    return x


def embeddings_forward_backward(x, model):
    out = model.transformer.embeddings(x)
    return out


def lm_head_forward_backward(x, model):
    out = model.lm_head(x)
    return out


if __name__ == '__main__':

    args = args_parser()
    if args.model == 'gpt2-small':
        config = GPT2Config()
    elif args.model == 'gpt2-medium':
        config = GPT2Config(n_embd=1024, n_head=16, n_layer=24)
    elif args.model == 'gpt2-large':
        config = GPT2Config(n_embd=1280, n_head=20, n_layer=36)
    else:
        raise 'not provide'

    config.use_flash_attn = True
    config.fused_bias_fc = False
    config.fused_mlp = False
    config.fused_dropout_add_ln = False
    config.residual_in_fp32 = True
    config.pad_vocab_size_multiple = 8
    model = GPTLMHeadModel(config).cuda()

    forward_result_path = os.path.join(args.save_path, f'{args.model}_forward.csv')
    if os.path.exists(forward_result_path):
        forward_result = pd.read_csv(forward_result_path, index_col=0)
    else:
        forward_result = pd.DataFrame()

    backward_result_path = os.path.join(args.save_path, f'{args.model}_backward.csv')
    if os.path.exists(backward_result_path):
        backward_result = pd.read_csv(backward_result_path, index_col=0)
    else:
        backward_result = pd.DataFrame()

    total_result_path = os.path.join(args.save_path, f'{args.model}_total.csv')
    if os.path.exists(total_result_path):
        total_result = pd.read_csv(total_result_path, index_col=0)
    else:
        total_result = pd.DataFrame()

    print('=========== start profiling ===========')
    print(f'model: {args.model}, batch size: {args.batch_size}, sequence legght: {args.seq_len}')

    x = torch.randint(0, 50257, (args.batch_size, args.seq_len)).to('cuda')
    label = copy.deepcopy(x)
    time_f, time_b = benchmark_fwd_bwd(
        full_model_forward_backward, x, label, model, repeats=args.repeats, desc="", verbose=False,
        amp=True, amp_dtype=torch.bfloat16)
    total_f = time_f[1]
    total_b = time_b[1]
    model_loss_f = total_f.mean
    model_loss_b = total_b.mean
    print('=========== full model + cross entropy===========')
    print(f'forward:{model_loss_f}, backward:{model_loss_b}')
    clean_up()
    del (x)
    del (model)
    model = GPTLMHeadModel(config).cuda()

    x = torch.randint(0, 50257, (args.batch_size, args.seq_len)).to('cuda')
    time_f, time_b = benchmark_fwd_bwd(
        model_forward_backward, x, model, repeats=args.repeats, desc="", verbose=False,
        amp=True, amp_dtype=torch.bfloat16)
    total_f = time_f[1]
    total_b = time_b[1]
    model_f = total_f.mean
    model_b = total_b.mean
    print('=========== full model ===========')
    print(f'forward:{model_f}, backward:{model_b}')
    clean_up()
    del (x)
    del (model)
    model = GPTLMHeadModel(config).cuda()

    x = torch.randint(0, 50257, (args.batch_size, args.seq_len)).to('cuda')
    time_f, time_b = benchmark_fwd_bwd(
        embeddings_forward_backward, x, model, repeats=args.repeats, desc="", verbose=False,
        amp=True, amp_dtype=torch.bfloat16)
    total_f = time_f[1]
    total_b = time_b[1]
    embedding_f = total_f.mean
    embedding_b = total_b.mean
    print('=========== embedding layers ===========')
    print(f'forward:{embedding_f}, backward:{embedding_b}')
    clean_up()
    del (x)
    del (model)
    model = GPTLMHeadModel(config).cuda()

    metrics = {'embedding': embedding_f, 'attention_block_lmhead': model_f - embedding_f,
               'cross_entropy': model_loss_f - model_f}
    df_new_row = pd.DataFrame(metrics, index=[args.batch_size])
    forward_result = pd.concat([forward_result, df_new_row])
    forward_result.to_csv(forward_result_path)

    metrics = {'embedding': embedding_b, 'attention_block_lmhead': model_b - embedding_b,
               'cross_entropy': model_loss_b - model_b}
    df_new_row = pd.DataFrame(metrics, index=[args.batch_size])
    backward_result = pd.concat([backward_result, df_new_row])
    backward_result.to_csv(backward_result_path)

    metrics = {'embedding': embedding_f + embedding_b,
               'attention_block_lmhead': (model_f + model_b) - (embedding_f + embedding_b),
               'cross_entropy': (model_loss_f + model_loss_b) - (model_f + model_b)}
    df_new_row = pd.DataFrame(metrics, index=[args.batch_size])
    total_result = pd.concat([total_result, df_new_row])
    total_result.to_csv(total_result_path)

    # x = torch.randn(args.batch_size, args.seq_len, 768).to('cuda').requires_grad_(True)
    # time_f, time_b = benchmark_fwd_bwd(
    #     attention_layer_forward_backward, x, model, repeats=args.repeats, desc="",
    #     verbose=False, amp=True, amp_dtype=torch.bfloat16)
    # total_f = time_f[1]
    # total_b = time_b[1]
    # total_f = total_f.mean
    # total_b = total_b.mean
    # print('=========== single attention layer ===========')
    # print(f'forward:{total_f}, backward:{total_b}')

    # x = torch.randn(args.batch_size, args.seq_len, 768).to('cuda').requires_grad_(True)
    # time_f, time_b = benchmark_fwd_bwd(
    #     attention_layers_forward_backward, x, model, repeats=args.repeats, desc="",
    #     verbose=False, amp=True, amp_dtype=torch.bfloat16)
    # total_f = time_f[1]
    # total_b = time_b[1]
    # total_f = total_f.mean
    # total_b = total_b.mean
    # print('=========== all attention layers ===========')
    # print(f'forward:{total_f}, backward:{total_b}')

    # x = torch.randn(args.batch_size, args.seq_len, 768).to('cuda').requires_grad_(True)
    # time_f, time_b = benchmark_fwd_bwd(
    #     lm_head_forward_backward, x, model, repeats=args.repeats, desc="",
    #     verbose=False, amp=True, amp_dtype=torch.bfloat16)
    # total_f = time_f[1]
    # total_b = time_b[1]
    # total_f = total_f.mean
    # total_b = total_b.mean
    # print('=========== lm head ===========')
    # print(f'forward:{total_f}, backward:{total_b}')

