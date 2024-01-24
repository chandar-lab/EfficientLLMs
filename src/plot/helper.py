import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantization import Quantized_Linear, Quantized_Conv2d
from transformers.pytorch_utils import Conv1D
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import mse_loss
from models import CausalGPT2_HF, CausalGPT2


def activation_hook(model, inputs):
    # add hook to record the min max value of the activation
    input_activation = {}
    output_activation = {}

    def add_range_recoder_hook(model):
        import functools
        def _record_range(self, x, y, module_name):
            x = x[0]
            input_activation[module_name] = x.detach()
            output_activation[module_name] = y.detach()

        all_hooks = []
        for name, m in model.named_modules():
            if isinstance(m, (
            nn.Linear, nn.LayerNorm, nn.ReLU, nn.GELU, nn.Embedding, Conv1D, Quantized_Linear, Quantized_Conv2d)):
                all_hooks.append(m.register_forward_hook(
                    functools.partial(_record_range, module_name=name)))
        return all_hooks

    hooks = add_range_recoder_hook(model)
    model(inputs)

    # remove hooks
    for h in hooks:
        h.remove()
    return input_activation, output_activation


def memory_breakdown_single_batch_summery(model, human_readable=True):
    activation_size = weight_size = grad_size = 0
    input_ = torch.randint(0, model.config.vocab_size, size=(1, model.config.n_positions))
    input_activation, _ = activation_hook(model, input_)
    for key in input_activation.keys():
        activation_size += input_activation[key].numel()
    activation_size = activation_size

    for name, p in model.named_parameters():
        weight_size += p.numel()
        if p.requires_grad:
            grad_size += p.numel()

    n_layer = model.config.n_layer
    n_head = model.config.n_head
    seq_lenght = model.config.n_positions
    attention_size = n_layer * n_head * seq_lenght * seq_lenght
    if human_readable:
        if min(weight_size, activation_size, attention_size) > 5e8:
            scale = 1e9
        else:
            scale = 1e6
        return {'weight': weight_size / scale, 'activation': activation_size / scale,
                'attention': attention_size / scale, 'grad': grad_size / scale}
    else:
        return {'weight': weight_size, 'activation': activation_size, 'attention': attention_size, 'grad': grad_size}


def memory_break_down_plot(model, show_attention=False):
    batch_size = [4., 16, 64, 256, 512]

    res = memory_breakdown_single_batch_summery(model, human_readable=False)
    weight = np.ones(len(batch_size)) * res['weight']
    activation = res['activation'] * np.array(batch_size)
    attention = res['attention'] * np.array(batch_size)
    grad = np.ones(len(batch_size)) * res['grad']
    adam_state = 2 * grad

    if show_attention:
        for i in range(len(batch_size)):
            sum = weight[i] + activation[i] + attention[i] + grad[i] + adam_state[i]
            weight[i] = weight[i] / sum
            activation[i] = activation[i] / sum
            attention[i] = attention[i] / sum
            grad[i] = grad[i] / sum
            adam_state[i] = adam_state[i] / sum

        fig = plt.figure(figsize=(8, 6))
        xticks = np.arange(len(batch_size))
        plt.bar(xticks, attention, label='attention')
        plt.bar(xticks, activation, bottom=attention, label='activation')
        plt.bar(xticks, weight, bottom=attention + activation, label='weight')
        plt.bar(xticks, grad, bottom=attention + activation + weight, label='grad')
        bar1 = plt.bar(xticks, adam_state, bottom=attention + activation + weight + grad, label='adam_state')
        bar = plt.xticks(xticks, np.array(batch_size, dtype=int), fontsize=12)
        plt.xlabel("batch size")
        plt.title("memory footprint")
        # Add counts above the two bar graphs
        for i, rect in enumerate(bar1):
            height = attention[i] + activation[i] / 2
            text = np.round(activation[i] * 100, 1)
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, f"%{text}", ha='center', va='bottom', fontsize=10)

        plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0, 0.15, 1, 1))
        plt.grid(axis='y')
    else:
        for i in range(len(batch_size)):
            sum = weight[i] + activation[i] + grad[i] + adam_state[i]
            weight[i] = weight[i] / sum
            activation[i] = activation[i] / sum
            grad[i] = grad[i] / sum
            adam_state[i] = adam_state[i] / sum

        fig = plt.figure(figsize=(8, 6))
        xticks = np.arange(len(batch_size))
        plt.bar(xticks, activation, label='activation')
        plt.bar(xticks, weight, bottom=activation, label='weight')
        plt.bar(xticks, grad, bottom=activation + weight, label='grad')
        bar1 = plt.bar(xticks, adam_state, bottom=activation + weight + grad, label='adam_state')
        bar = plt.xticks(xticks, np.array(batch_size, dtype=int), fontsize=12)
        plt.xlabel("batch size")
        plt.title("memory footprint")
        # Add counts above the two bar graphs
        for i, rect in enumerate(bar1):
            height = activation[i] / 2
            text = np.round(activation[i] * 100, 1)
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, f"%{text}", ha='center', va='bottom', fontsize=10)

        plt.legend(loc='upper center', ncol=5, bbox_to_anchor=(0, 0.15, 1, 1))
        plt.grid(axis='y')


def plot_mse_layer(c_attn, c_proj, mlp_c_fc, mlp_c_proj, save_path: str = './save',
                   title: str = 'W8_sym_per-tensor'):
    # set plot style
    fsize = 22
    tsize = 20
    tdir = 'in'
    major = 1.0
    minor = 1.0
    style = 'default'
    plt.style.use(style)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    plt.rcParams['lines.linewidth'] = 2
    xsize = 8
    ysize = 4
    colors = ['#003790', '#6fc2db', '#ea6372', '#93003a']

    fig, axes = plt.subplots(figsize=(xsize, ysize))
    xticks = np.arange(len(mlp_c_proj))
    width = .1
    axes.bar(xticks - (2 * width), c_attn, width=width, color=colors[0], label='c_attn')
    axes.bar(xticks - width, c_proj, width=width, color=colors[1], label='c_proj')
    axes.bar(xticks, mlp_c_fc, width=width, color=colors[2], label='fc1')
    axes.bar(xticks + width, mlp_c_proj, width=width, color=colors[3], label='fc2')
    axes.set_xticks(xticks)
    axes.set_xticklabels([str(int(_)) for _ in xticks])
    axes.set_xlabel("layer index")
    axes.set_ylabel("error")
    fig.suptitle(title, fontsize=14)

    axes.legend(loc='center right', ncol=1, bbox_to_anchor=(0, 0.15, 1.3, 1))
    axes.grid(axis='y', linestyle='-.', linewidth=1.)
    plt.savefig(os.path.join(save_path, f'{title}.png'), dpi=300, pad_inches=.1, bbox_inches='tight')


def extract_mse_between_layers(model, output_activation: dict, output_activation_quant: dict):
    n_layer = model.config.n_layer

    c_attn = np.zeros(n_layer)
    c_proj = np.zeros(n_layer)
    mlp_c_fc = np.zeros(n_layer)
    mlp_c_proj = np.zeros(n_layer)

    if isinstance(model, CausalGPT2):
        for i in range(n_layer):
            name = f'transformer.layers.{i}.mixer.Wqkv'
            c_attn[i] = mse_loss(output_activation_quant[name], output_activation[name]).item()
            name = f'transformer.layers.{i}.mixer.out_proj'
            c_proj[i] = mse_loss(output_activation_quant[name], output_activation[name]).item()
            name = f'transformer.layers.{i}.mlp.fc1'
            mlp_c_fc[i] = mse_loss(output_activation_quant[name], output_activation[name]).item()
            name = f'transformer.layers.{i}.mlp.fc2'
            mlp_c_proj[i] = mse_loss(output_activation_quant[name], output_activation[name]).item()

    elif isinstance(model, CausalGPT2_HF):
        for i in range(n_layer):
            name = f'transformer.h.{i}.attn.c_attn'
            c_attn[i] = mse_loss(output_activation_quant[name], output_activation[name]).item()
            name = f'transformer.h.{i}.attn.c_proj'
            c_proj[i] = mse_loss(output_activation_quant[name], output_activation[name]).item()
            name = f'transformer.h.{i}.mlp.c_fc'
            mlp_c_fc[i] = mse_loss(output_activation_quant[name], output_activation[name]).item()
            name = f'transformer.h.{i}.mlp.c_proj'
            mlp_c_proj[i] = mse_loss(output_activation_quant[name], output_activation[name]).item()

    else:
        raise NotImplementedError

    return c_attn, c_proj, mlp_c_fc, mlp_c_proj


def plot_eval_on_checkpoints(metrics, save_path: str = None):
    # plot style:
    fsize = 22
    tsize = 20
    tdir = 'in'
    major = 1.0
    minor = 1.0
    style = 'default'
    xsize = 8
    ysize = 5
    plt.style.use(style)
    # plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    plt.rcParams['lines.linewidth'] = 2

    assert 'iters' in metrics
    dash_style = ['-o', '-s', '-x', '-d']
    colors = ['#003790', '#6fc2db', '#ea6372', '#93003a']
    num_metrics = len(metrics.keys()) - 1
    iters = np.array(metrics.pop('iters'))/10000
    fig, axes = plt.subplots(1, num_metrics, figsize=(num_metrics * xsize, 1 * ysize))
    for i, key in enumerate(metrics.keys()):
        axes[i].plot(iters, metrics[key], dash_style[i], color=colors[i], alpha=1.0)
        axes[i].set_xlabel('Training Iteration')
        axes[i].set_title(key)
        axes[i].text(1.12, -0.12, '(1e4)', transform=axes[i].transAxes,
                     ha='right', va='bottom', fontsize=fsize, color='black')
        axes[i].grid()
    plt.savefig(os.path.join(save_path, 'eval_on_checkpoints.pdf'), dpi=300, pad_inches=.1, bbox_inches='tight')
