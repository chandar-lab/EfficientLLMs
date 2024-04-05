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
                    nn.Linear, nn.LayerNorm, nn.ReLU, nn.GELU, nn.Embedding, Conv1D, Quantized_Linear,
                    Quantized_Conv2d)):
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
    fsize = 18
    tsize = 16
    tdir = 'in'
    major = 1.0
    minor = 1.0
    style = 'default'
    xsize = 12
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

    def format_ticks(tick_positions):
        return [f'{int(pos / 1000)}k' if pos > 0 else '0' for pos in tick_positions]

    assert 'iters' in metrics
    dash_style = ['-o', '-s', '-x', '-d']
    colors = ['#003790', '#6fc2db', '#ea6372', '#93003a']
    num_metrics = len(metrics.keys()) - 1
    iters = np.array(metrics.pop('iters'))
    fig, axes = plt.subplots(num_metrics, 1, figsize=(1 * xsize, num_metrics * ysize), gridspec_kw={'hspace': 0.5})
    for i, key in enumerate(metrics.keys()):
        axes[i].plot(iters, metrics[key], dash_style[i], color=colors[i], alpha=1.0)
        axes[i].set_xlabel('Training Iteration')
        axes[i].set_title(key)
        tick_positions = np.linspace(0, max(iters), 6)
        axes[i].set_xticks(tick_positions)
        axes[i].set_xticklabels(format_ticks(tick_positions))
        axes[i].grid()
        if key == 'eval_loss':
            axes[i].set_ylim(min(metrics[key]) - 0.1, max(metrics[key]) + 0.1)
    plt.savefig(os.path.join(save_path, 'eval_on_checkpoints.pdf'), dpi=300, pad_inches=.1, bbox_inches='tight')


def plot_activations_histogram(input_activation, layer=0, attention_only=False, n_head=12):
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
    ysize = 5.5
    num_bins = 256

    list_of_activations = ['norm1', 'mixer.Wqkv', 'mixer.out_proj', 'norm2', 'mlp.fc1', 'mlp.fc2']
    layer_name = f'transformer.layers.{layer}'

    fig, axes = plt.subplots(1, len(list_of_activations), figsize=(len(list_of_activations) * xsize, 1 * ysize))
    for i, act in enumerate(list_of_activations):
        act_ = input_activation[f'{layer_name}.{act}'].detach().cpu().float().numpy().reshape(-1)
        counts, bins = np.histogram(act_, num_bins)
        counts = counts / len(act_)
        width_ = (max(bins) - min(bins)) / num_bins
        axes[i].bar(bins[:-1], counts, width=width_, alpha=0.9)
        xticks = np.linspace(min(bins), max(bins), 5)
        xticks = np.round(xticks, 3)
        axes[i].set_xticks(xticks)
        axes[i].grid()
        axes[i].set_yscale('log')
        axes[i].set_title(act)
    fig.suptitle(f' Input Activations of Layer {layer} ', fontsize=30, y=1.05)



########################################################################

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import wandb

def get_run_history(run_name='gpt2_A4_sym_per_column_9b08f23b'):
    api = wandb.Api()
    runs = api.runs("efficient_llm/ablation study 2")
    # run_history = {'eval_loss':[], 'train_global_step':[], 'train_loss':[], 'grad_norm': []}
    run_history = {'eval_loss':[], 'train_global_step':[], 'grad_norm': []}
    for run in runs:
        if run.displayName == run_name:
            for k in run.scan_history():
                if run.tags:
                    if 'old' in run.tags:
                        continue
                if 'eval/loss' in k.keys() and k['eval/loss'] is not None:
                    if run_name == 'gpt2_A4_sym_per_column_9b08f23b' and k['train/global_step']>37000:
                        continue
                    if run_name == 'gpt2_A4_sym_per_tensor_b6b01382' and k['train/global_step']>20000:
                        continue
                    run_history['eval_loss'].append(k['eval/loss'])
                    run_history['train_global_step'].append(k['train/global_step'])
                    # run_history['train_loss'].append(k['train/loss'])
                    run_history['grad_norm'].append(k['model/grad_norm'])
    run_history_ = {}
    for k,v in run_history.items():
        run_history_[k] = np.array(v)
    return run_history_

import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def save_runs_from_wandb(run_names: str = 'gpt2_A4_sym_per_column_9b08f23b'):
    run_names = [f.strip() for f in run_names.split(",")]
    for i, run_name in enumerate(run_names):
        run_history = get_run_history(run_name)
        df = pd.DataFrame(run_history)
        df.to_csv(os.path.join('../save_new', run_name, 'run_history.csv'), index=False)

# save_runs_from_wandb('gpt2_G8_split_quant_sym_per_tensor_91ca4778, gpt2_G8_split_quant_sym_per_column_06ae6da0')
def plot_runs_from_wandb(run_names=['gpt2_A4_sym_per_column_9b08f23b']):
    set_plot_style(linewidth=2.)
    xsize = 8
    ysize = 5.7

    def return_true_label(label):
        color_plater = ['#00429d', '#307ab5', '#4cb6bf', '#ffa482', '#dc4c5d', '#93003a']
        if 'asym' in label:
            res += ' asymmetric'
            dash = '--'
        else:
            dash = '-'
        if '8' in label:
            if 'per_tensor' in label:
                res = '8bit per-tensor'
                color = color_plater[0]
            elif 'per_column' in label:
                res = '8bit per-channel'
                color = color_plater[1]
            else:
                res = '8bit per-token'
                color = color_plater[2]
        elif '4' in label:
            if 'per_tensor' in label:
                res = '4bit per-tensor'
                color = color_plater[5]
            elif 'per_column' in label:
                res = '4bit per-channel'
                color = color_plater[4]
            else:
                res = '4bit per-token'
                color = color_plater[3]
        else:
            res = 'baseline'
            color = '#ff69b4'
        return res, color, dash

    def singel_plot(axes, run_names):
        max_len = 0
        for i, run_name in enumerate(run_names):
            run_history = pd.read_csv(os.path.join('../save_new', run_name, 'run_history.csv'))
            idx = run_history['train_global_step'].argsort()
            label = run_name[:run_name.rfind('_')]
            label = label.replace("per_column", "per_token")
            label, c, dash = return_true_label(label)
            axes.plot(run_history['train_global_step'][idx], run_history['eval_loss'][idx], dash, color=c, alpha=.85,
                      label=label)
            max_len = max(max_len, max(run_history['train_global_step']))

        def format_ticks(tick_positions):
            return [f'{int(pos / 1000)}k' if pos > 0 else '0' for pos in tick_positions]

        # Get x-axis range and set custom ticks
        print(max_len)
        x_range = (0, max_len)  # Adjust the range as needed
        tick_positions = np.linspace(x_range[0], x_range[1], 6)
        axes.set_xticks(tick_positions)
        axes.set_xticklabels(format_ticks(tick_positions))

        axes.set_xlabel('Training Iteration')
        axes.set_ylabel('Eval Loss')
        axes.grid()

    fig, axes = plt.subplots(1, len(run_names), figsize=(xsize * len(run_names), ysize))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    all_handles_labels = []  # List to store all handles and labels

    for i, run_group in enumerate(run_names):
        run_name = [f.strip() for f in run_group.split(",")]
        singel_plot(axes[i], run_name)
        # Collect handles and labels from each subplot
        handles, labels = axes[i].get_legend_handles_labels()
        all_handles_labels.extend(zip(handles, labels))

    # Remove duplicates from the list
    unique_handles_labels = list(dict.fromkeys(all_handles_labels))

    # Unzip the handles and labels
    unique_handles, unique_labels = zip(*unique_handles_labels)

    # Create a single legend with the unique handles and labels
    fig.legend(unique_handles, unique_labels, loc='lower center', ncol=4, bbox_to_anchor=(0, -0.05, 1, 1),
               bbox_transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.3)  # Adjust bottom to accommodate the legend

    # plt.savefig(os.path.join('../save_new', 'wandb_logs.pdf'), dpi=300, pad_inches=.1, bbox_inches='tight')


import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from plot.histogram import set_plot_style


def plot_runs_from_wandb(run_names='gpt2_A4_sym_per_column_9b08f23b', zoom=6, y2=3.1, save_name=None,
                         downstream_result=None):
    set_plot_style(linewidth=2.5, fsize=12, tsize=14)
    color_plater = {'8bit per-tensor': '#4cb6bf',
                    '8bit per-channel': '#307ab5',
                    '8bit per-token': '#00429d',
                    '4bit per-tensor': '#ffa482',
                    '4bit per-channel': '#dc4c5d',
                    '4bit per-token': '#93003a'
                    }
    xsize = 6.5 * 1.2
    ysize = 3.6 * 1.2

    def return_true_label(label):
        if '8' in label:
            if 'per_tensor' in label:
                res = '8bit per-tensor'
                color = color_plater[res]
            elif 'per_column' in label:
                res = '8bit per-channel'
                color = color_plater[res]
            else:
                res = '8bit per-token'
                color = color_plater[res]
        elif '4' in label:
            if 'per_tensor' in label:
                res = '4bit per-tensor'
                color = color_plater[res]
            elif 'per_column' in label:
                res = '4bit per-channel'
                color = color_plater[res]
            else:
                res = '4bit per-token'
                color = color_plater[res]
        else:
            res = 'baseline'
            color = '#ff69b4'  # '#662E7D'
        if 'asym' in label:
            res += ' asymmetric'
            dash = '--'
        else:
            dash = '-'
        return res, color, dash

    fig, axes = plt.subplots(1, 2, figsize=(2 * xsize, ysize))
    run_names = [f.strip() for f in run_names.split(",")]
    max_len = 0
    for i, run_name in enumerate(run_names):
        run_history = pd.read_csv(os.path.join('../save_new', run_name, 'run_history.csv'))
        if run_name == 'gpt2__9959030e':
            print('remove ourlier in gpt2__9959030e')
            outlier_indices = run_history['eval_loss'] > 5.5
            run_history.loc[outlier_indices, 'eval_loss'] = run_history['eval_loss'].shift(1)[outlier_indices]
        idx = run_history['train_global_step'].argsort()
        label = run_name[:run_name.rfind('_')]
        if 'A4' in label and 'per_token' in label:
            print('change lable:', label)
            label = label
            label = label.replace("per_token", "per_column")
            print('-->:', label, 'because of 1th if')
        elif ('W4' not in label and 'W8' not in label):
            print('change lable:', label)
            label = label.replace("per_column", "per_token")
            print('-->:', label, 'because of 2th if')
        if 'S1' in label:
            print('change lable:', label)
            label = label.replace("per_token", "per_column")
            print('-->:', label, 'because of 3th if')

        label, c, dash = return_true_label(label)
        axes[0].plot(run_history['train_global_step'][idx], run_history['eval_loss'][idx], dash, color=c, alpha=.85,
                     label=label)
        axes[0].set_ylim(top=5)
        max_len = max(max_len, max(run_history['train_global_step']))

    def format_ticks(tick_positions):
        return [f'{int(pos / 1000)}k' if pos > 0 else '0' for pos in tick_positions]

    # print(max_len)
    x_range = (0, max_len)  # Adjust the range as needed
    tick_positions = np.linspace(x_range[0], x_range[1], 6)
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(format_ticks(tick_positions))

    axes[0].set_xlabel('Training Iteration')
    axes[0].set_ylabel('Eval Loss')
    axes[0].grid()
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(0.9, .5, 1, 0),
    #            bbox_transform=fig.transFigure)
    print(labels)
    fig.legend(handles, labels, loc='lower center', ncol=len(run_names), bbox_to_anchor=(0.0, .9, 1, 1),
               bbox_transform=fig.transFigure)

    if zoom:
        x1, x2 = 270000, 300000
        y1 = 2.85
        # Make the zoom-in plot:
        axins = zoomed_inset_axes(axes[0], zoom, loc=1)  # zoom = 2
        for i, run_name in enumerate(run_names):
            run_history = pd.read_csv(os.path.join('../save_new', run_name, 'run_history.csv'))
            label = run_name[:run_name.rfind('_')]
            # if 'A4' in label and 'per_token' in label:
            #     pass
            # elif ('W4' not in label and 'W8' not in label):
            #     print('change lable:' label)
            #     label = label.replace("per_column", "per_token")
            #     print('-->:', label)
            # if 'S1' in label:
            #     print(label)
            #     label = label.replace("per_token", "per_column")
            label, c, dash = return_true_label(label)
            idx = run_history['train_global_step'].argsort()
            axins.plot(run_history['train_global_step'][idx], run_history['eval_loss'][idx], dash, color=c, alpha=.85)

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.grid()
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(axes[0], axins, loc1=1, loc2=2, fc="none", ec="0.5")

    ##############################################################################
    base_line = {'WikiText2': 34.32, 'WikiText103': 39.94, 'Lambada': 34.8, 'PTB': 35.13, '1BW': 44.03}
    perplexities = list(downstream_result.keys())
    quantization_types = list(downstream_result['WikiText2'].keys())
    hatch_pattern = '///'
    bar_width = 0.14
    for i, perplexity in enumerate(perplexities):
        for j, quantization_type in enumerate(quantization_types):
            if downstream_result[perplexity][quantization_type] is None:
                continue
            axes[1].bar(
                i + j * bar_width,
                downstream_result[perplexity][quantization_type],
                width=bar_width,
                label=f'{perplexity} - {quantization_type}',
                color=color_plater[quantization_type],
                edgecolor='black',
            )
        axes[1].bar(
            i + (j + 1) * bar_width,
            base_line[perplexity],
            width=bar_width,
            label=f'{perplexity} - base_line',
            color='#ff69b4',
            edgecolor='black',
        )
    # Set x-axis ticks and labels
    axes[1].set_xticks(np.arange(len(perplexities)) + (len(quantization_types) - 1) * bar_width / 2)
    axes[1].set_xticklabels(perplexities)

    # Set labels and legend
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Perplexity')  # Replace 'Your Y-Axis Label' with the appropriate label

    if save_name is not None:
        plt.savefig(os.path.join('../save_new', f'{save_name}_wandb_logs.pdf'), dpi=300, pad_inches=.1,
                    bbox_inches='tight')
