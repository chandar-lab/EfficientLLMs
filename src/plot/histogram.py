import torch
from transformers.pytorch_utils import Conv1D
import matplotlib.pyplot as plt
import os
import numpy as np
from plot import activation_hook
import math
import seaborn as sns
import copy
import torch.nn.functional as F
import transformers
from models import dequant_model


def set_plot_style(
        fsize: int = 14,
        tsize: int = 10,
        tdir: str = 'in',
        major: float = 1.0,
        minor: float = 1.0,
        style: str = 'default',
        use_latex_format: bool = False,
        linewidth: float = 2.0,
):
    plt.style.use(style)
    plt.rcParams['text.usetex'] = use_latex_format
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = tsize
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = major
    plt.rcParams['ytick.minor.size'] = minor
    plt.rcParams['lines.linewidth'] = linewidth


# def get_intermediate_activations(model, tokenizer, prompt, device=torch.device('cuda')):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     input_activation, output_activation = activation_hook(model, inputs['input_ids'])
#     return input_activation, output_activation

def plot_intermediate_activation_histogram_3d(activation, layer=0, layer_type: str = 'mixer.Wqkv',
                                              title: str = None, save_path: str = None):
    set_plot_style()
    if layer_type == 'lm_head':
        X = activation[f'lm_head'].mean(0)
    else:
        X = activation[f'transformer.layers.{layer}.{layer_type}'].mean(0)

    data = X.detach().abs().cpu().float().numpy()
    T, C = data.shape
    token_indices, channel_indices = np.meshgrid(range(T), range(C), indexing='ij')

    fig = plt.figure(figsize=(9, 7))
    ax = plt.axes(projection='3d')
    ax.plot_surface(token_indices, channel_indices, data, cmap='RdBu_r', alpha=.7, linewidth=0)

    ax.set_xlabel('Tokens', labelpad=8)
    ax.set_ylabel('Channels', labelpad=8)
    # ax.set_zlabel('Abs values', fontsize=12)

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # Remove z-axis tick labels
    # Remove z-axis
    ax.w_zaxis.line.set_lw(0)
    ax.set_zticks([])
    if title is not None:
        fig.suptitle(f'{title}', y=0.1)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'intermediate_activation_histogram_3d_{title}.png'), dpi=300,
                    pad_inches=.1, bbox_inches='tight')


def plot_intermediate_activation_histogram_3d_wireframe(model, tokenizer, prompt, layer=0,
                                                        layer_type: str = 'mixer.Wqkv',
                                                        title: str = None, save_path: str = None):
    set_plot_style()
    device = torch.device('cuda')
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_activation, output_activation = activation_hook(model, inputs['input_ids'])
    if layer_type == 'lm_head':
        X = input_activation[f'lm_head'][0]
    else:
        X = input_activation[f'transformer.layers.{layer}.{layer_type}'][0]

    data = X.detach().abs().cpu().float().numpy()
    T, C = data.shape
    xdata = np.array([np.linspace(0, T - 1, T) for i in range(C)])
    ydata = np.array([np.ones(T) * i for i in range(C)])

    fig = plt.figure(figsize=(14, 6))
    fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(wspace=0.13)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_wireframe(xdata, ydata, data.T, rstride=0, color="royalblue", linewidth=1.5)

    # ax.set_xlabel('Tokens')
    ax.set_ylabel('Channels', labelpad=8)
    xtick_labels = [tokenizer.decode(_) for _ in inputs['input_ids'][0]]
    xtick_labels = ['\\n' if _ == '\n' else _ for _ in xtick_labels]
    ax.set_xticks(xdata[0])
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=10)

    if title is not None:
        fig.suptitle(f'{title}', y=0.1)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'intermediate_activation_histogram_3d_wireframe_{title}.png'), dpi=300,
                    pad_inches=.1, bbox_inches='tight')


from matplotlib.animation import FuncAnimation


def animated_intermediate_activation_histogram_3d_wireframe(model, tokenizer, prompt, layer_type='mixer.Wqkv',
                                                            save_path=None):
    set_plot_style()
    device = torch.device('cuda')
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    fig = plt.figure(figsize=(14, 6))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.13)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    def update(frame):
        ax.cla()  # Clear previous plot
        layer = frame
        input_activation, output_activation = activation_hook(model, inputs['input_ids'])
        if layer_type == 'lm_head':
            X = input_activation[f'lm_head'][0]
        else:
            X = input_activation[f'transformer.layers.{layer}.{layer_type}'][0]

        data = X.detach().abs().cpu().float().numpy()
        T, C = data.shape
        xdata = np.array([np.linspace(0, T - 1, T) for i in range(C)])
        ydata = np.array([np.ones(T) * i for i in range(C)])

        ax.plot_wireframe(xdata, ydata, data.T, rstride=0, color="royalblue", linewidth=1.)

        ax.set_ylabel('Channels', labelpad=8)
        xtick_labels = [tokenizer.decode(_) for _ in inputs['input_ids'][0]]
        xtick_labels = ['\\n' if _ == '\n' else _ for _ in xtick_labels]
        ax.set_xticks(xdata[0])
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=10)
        ax.set_title(f'layer {layer}')

    anim = FuncAnimation(fig, update, frames=range(model.config.n_layer), repeat=False)

    if save_path is not None:
        anim.save(os.path.join(save_path, 'intermediate_activation_histogram_3d.gif'),
                  writer='pillow', fps=2, dpi=300, savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1})
    else:
        plt.show()

def plot_intermediate_activation_histogram_2d_outlier(activation, layer=0, layer_type: str = 'mixer.Wqkv',
                                                      title: str = None, save_path: str = None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    set_plot_style()
    if layer_type == 'lm_head':
        X = activation['lm_head'].mean(0).abs().detach().cpu().float()
    else:
        X = activation[f'transformer.layers.{layer}.{layer_type}'].mean(0).abs().detach().cpu().float()

    # get out in channels
    x_col = X.mean(0)
    outlier_thershold_col = x_col.mean() + 6 * x_col.std()
    outliser_index_col = x_col > outlier_thershold_col
    outliser_index_col = torch.nonzero(outliser_index_col).squeeze()

    # get outliers in tokens
    x_token = X.mean(1)
    outlier_thershold_token = x_token.mean() + 6 * x_token.std()
    outliser_index_token = x_token > outlier_thershold_token
    outliser_index_token = torch.nonzero(outliser_index_token).squeeze()
    if outliser_index_token.numel() == 1:
        outliser_index_token = torch.tensor([outliser_index_token.tolist()])

    mean_abs_histogram = torch.zeros((len(x_token), len(x_col)))
    if outliser_index_col.numel() > 0:
        print(
            f'channel ourlier index:\n{outliser_index_col.tolist()}, values: {x_col[outliser_index_col].tolist()}')
        mean_abs_histogram[:, outliser_index_col] = x_col[outliser_index_col]
    if outliser_index_token.numel() > 0:
        print(
            f'\ntoken ourlier index:\n{outliser_index_token.tolist()}, values: {x_token[outliser_index_token].tolist()}')
        for t in outliser_index_token:
            mean_abs_histogram[t, :] = x_token[t]

    fig, axe = plt.subplots(figsize=(8, 5))
    img = axe.imshow(mean_abs_histogram, cmap='RdBu_r', aspect='auto', extent=[0, len(x_col), 0, len(x_token)])
    # Add colorbar
    divider = make_axes_locatable(axe)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax)
    axe.set_xticks(outliser_index_col)
    axe.set_xticklabels(outliser_index_col.tolist(), rotation=45, ha='right')
    if outliser_index_token.numel() > 1:
        axe.set_yticks(outliser_index_token)
        axe.set_yticklabels(outliser_index_token.tolist())
    axe.set_xlabel('Channels')
    axe.set_ylabel('Tokens')
    axe.set_title('Mean Absolute value')

    if title is not None:
        fig.suptitle(f'{title}', y=0.1)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'intermediate_activation_histogram_2d_outlier_{title}.png'), dpi=300,
                    pad_inches=.1, bbox_inches='tight')


from mpl_toolkits.axes_grid1 import make_axes_locatable


def animated_intermediate_activation_histogram_2d_outlier_gif(activation, layer_type='mixer.Wqkv', num_layers=12,
                                                              save_path=None):
    set_plot_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.tight_layout()

    def update(frame):
        ax.cla()  # Clear previous plot
        layer = frame

        if layer_type == 'lm_head':
            X = activation['lm_head'].mean(0).abs().detach().cpu().float()
        else:
            X = activation[f'transformer.layers.{layer}.{layer_type}'].mean(0).abs().detach().cpu().float()

        # get out in channels
        x_col = X.mean(0)
        outlier_thershold_col = x_col.mean() + 6 * x_col.std()
        outliser_index_col = x_col > outlier_thershold_col
        outliser_index_col = torch.nonzero(outliser_index_col).squeeze()

        # get outliers in tokens
        x_token = X.mean(1)
        outlier_thershold_token = x_token.mean() + 6 * x_token.std()
        outliser_index_token = x_token > outlier_thershold_token
        outliser_index_token = torch.nonzero(outliser_index_token).squeeze()
        if outliser_index_token.numel() == 1:
            outliser_index_token = torch.tensor([outliser_index_token.tolist()])

        mean_abs_histogram = torch.zeros((len(x_token), len(x_col)))
        if outliser_index_col.numel() > 0:
            print(
                f'channel ourlier index:\n{outliser_index_col.tolist()}, values: {x_col[outliser_index_col].tolist()}')
            mean_abs_histogram[:, outliser_index_col] = x_col[outliser_index_col]
        if outliser_index_token.numel() > 0:
            print(
                f'\ntoken ourlier index:\n{outliser_index_token.tolist()}, values: {x_token[outliser_index_token].tolist()}')
            for t in outliser_index_token:
                mean_abs_histogram[t, :] = x_token[t]

        img = ax.imshow(mean_abs_histogram, cmap='RdBu_r', aspect='auto', extent=[0, len(x_col), 0, len(x_token)])

        # Remove previous colorbar before adding a new one
        cbar_ax = fig.colorbar(img, ax=ax)
        cbar_ax.remove()

        ax.set_xticks(outliser_index_col)
        ax.set_xticklabels(outliser_index_col.tolist(), rotation=45, ha='right')

        if outliser_index_token.numel() > 1:
            ax.set_yticks(outliser_index_token)
            ax.set_yticklabels(outliser_index_token.tolist())

        ax.set_xlabel('Channels')
        ax.set_ylabel('Tokens')
        ax.set_title('Mean Absolute value - Layer {}'.format(layer))

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(img, cax=cax)

    anim = FuncAnimation(fig, update, frames=range(num_layers), repeat=False)

    if save_path is not None:
        anim.save(os.path.join(save_path, 'intermediate_activation_histogram_2d_outlier.gif'),
                  writer='pillow', fps=2, dpi=300, savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1})
    else:
        plt.show()


def return_attention(output_activation, layer=0, num_head=12):
    B, S, DH = output_activation[f'transformer.layers.{layer}.mixer.Wqkv'].shape
    H = num_head
    qkv = output_activation[f'transformer.layers.{layer}.mixer.Wqkv'].reshape(B, S, 3, H, DH // (3 * H))
    batch_size, seqlen = qkv.shape[0], qkv.shape[1]
    q, k, v = qkv.unbind(dim=2)
    softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    causal_mask = torch.triu(
        torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
    )
    scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
    return attention


def plot_attentions(output_activation, layer=0, mean=True):
    atten = return_attention(output_activation, layer).detach().cpu().float().numpy()[0]
    if mean:
        atten = atten.mean(0)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4.75))
        fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
        plt.subplots_adjust(wspace=0.15)
        sns.heatmap(atten, square=True, ax=ax, cmap="YlGnBu", cbar_kws={"shrink": 1.0, "pad": 0.01, "aspect": 50})

        ax.set_facecolor("whitesmoke")
        cax = ax.figure.axes[-1]
        cax.tick_params(labelsize=18)

        ax.tick_params(axis='x', which='major')
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])
        ax.tick_params(left=False, bottom=False)

    else:
        num_head = atten.shape[0]
        fig, axes = plt.subplots(nrows=1, ncols=num_head, figsize=(4 * num_head, 4))
        fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
        plt.subplots_adjust(wspace=0.15)
        for i in range(num_head):
            sns.heatmap(atten[i], square=True, ax=axes[i], cmap="YlGnBu",
                        cbar_kws={"shrink": 1.0, "pad": 0.01, "aspect": 50})
            axes[i].set_facecolor("whitesmoke")
            cax = axes[i].figure.axes[-1]
            cax.tick_params(labelsize=12)
            axes[i].tick_params(axis='x', which='major')
            axes[i].set(xticklabels=[])
            axes[i].set(yticklabels=[])
            axes[i].tick_params(left=False, bottom=False)
            axes[i].set_title(f'head {i}')


def calculate_gradients(model, data):
    import transformers
    model.zero_grad()
    out = model(**data)
    out.loss.backward()
    gradients = {}
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Linear, transformers.pytorch_utils.Conv1D)):
            gradients[name] = m.weight.grad.detach().abs().cpu().float()

    return gradients

def plot_grad_histogram_2d(model, data, layer: int = 0, layer_type='mixer.Wqkv', num_split=1):
    grads = calculate_gradients(model, data)
    if layer_type == 'lm_head':
        data = grads[f'lm_head']
    else:
        data = grads[f'transformer.layers.{layer}.{layer_type}']

    set_plot_style(fsize=14, tsize=8)
    if num_split == 1:
        data = data.numpy()
        dim0, dim1 = data.shape
        dim0_indices, dim1_indices = np.meshgrid(range(dim0), range(dim1), indexing='ij')

        fig = plt.figure(figsize=(9, 7))
        ax = plt.axes(projection='3d')
        ax.plot_surface(dim0_indices, dim1_indices, data, cmap='RdBu_r', alpha=.7, linewidth=0)

        ax.set_xlabel('dim0', labelpad=10)
        ax.set_ylabel('dim1', labelpad=10)
        ax.set_title(f'layer {layer} {layer_type}')
    else:
        rows_per_split = data.size(0) // num_split
        splitted_data = data.split(rows_per_split, dim=0)
        fig = plt.figure(figsize=(9 * num_split, 7))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.13)
        axes = [fig.add_subplot(1, num_split, i + 1, projection='3d') for i in range(num_split)]
        for i, d in enumerate(splitted_data):
            # d = d.numpy()
            # dim0, dim1 = d.shape
            # dim0_indices, dim1_indices = np.meshgrid(range(dim0), range(dim1), indexing='ij')
            # axes[i].plot_surface(dim0_indices, dim1_indices, d, cmap='RdBu_r', alpha=.7, linewidth=0)
            dim0, dim1 = d.shape
            dim0_indices, dim1_indices = torch.meshgrid(torch.arange(dim0) + i * dim0, torch.arange(dim1))
            axes[i].plot_surface(dim0_indices.numpy(), dim1_indices.numpy(), d.numpy(), cmap='RdBu_r', alpha=.7,
                                 linewidth=0)

            axes[i].set_xlabel('dim0', labelpad=10)
            axes[i].set_ylabel('dim1', labelpad=10)
        fig.suptitle(f'Layer {layer} {layer_type}', y=0.95)


def check_grad_per_layer(experiment, metric='cos_sim', title: str = None):

    model = experiment.trainer.accelerator.prepare_model(experiment.trainer.model, evaluation_mode=True)
    model.eval()
    data_loader = experiment.trainer.get_train_dataloader()
    data = next(iter(data_loader))

    model_dequant = copy.deepcopy(model)
    dequant_model(model_dequant)

    print('check the forward path')
    out = model(**data)
    out_real = model_dequant(**data)
    print('logits:', torch.allclose(out.logits, out_real.logits))

    out.loss.backward()
    out_real.loss.backward()

    cos_sim = {}
    l2_norm = {}
    l1_norm = {}
    index = {}

    i = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.pytorch_utils.Conv1D):
            m_dequant = model_dequant.get_submodule(name)
            grad_model = m.weight.grad
            grad_model_dequant = m_dequant.weight.grad

            cos_sim[name] = F.cosine_similarity(grad_model.view(-1), grad_model_dequant.view(-1), dim=0).item()
            l2_norm[name] = torch.norm(grad_model - grad_model_dequant, p=2).item()
            l1_norm[name] = torch.norm(grad_model - grad_model_dequant, p=1).item()
            index[name] = i
            i += 1

    label_mapping = {
        'Wqkv': 'Wqkv',
        'out_proj': 'Out Proj',
        'fc1': 'FC1',
        'fc2': 'FC2',
    }

    # Initialize lists for each layer type
    layer_types = ['Wqkv', 'out_proj', 'fc1', 'fc2']
    layer_data = {layer: [] for layer in layer_types}
    layer_indices = {layer: [] for layer in layer_types}

    if metric == 'cos_sim':
        for layer in layer_types:
            for key, value in cos_sim.items():
                if layer in key:
                    layer_indices[layer].append(index[key])
                    layer_data[layer].append(value)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(cos_sim.values(), '--', c='gray', label='Overall')
        for layer in layer_types:
            ax.plot(layer_indices[layer], layer_data[layer], 's', label=label_mapping[layer])
        ax.plot(index['lm_head'], cos_sim['lm_head'], 's', label='lm_head')
        custom_xticks = [i * len(layer_types) + 1 for i in range(len(cos_sim) // len(layer_types))]
        ax.set_xticks(custom_xticks)
        ax.set_xticklabels(range(len(cos_sim) // len(layer_types)))
        ax.set_xlabel('Layer index')
        ax.set_ylabel('Cosine Similarity')
    elif metric == 'l2_norm':
        for layer in layer_types:
            for key, value in l2_norm.items():
                if layer in key:
                    layer_indices[layer].append(index[key])
                    layer_data[layer].append(value)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(l2_norm.values(), '--', c='gray', label='Overall')
        for layer in layer_types:
            ax.plot(layer_indices[layer], layer_data[layer], 's', label=label_mapping[layer])
        ax.plot(index['lm_head'], l2_norm['lm_head'], 's', label='lm_head')
        custom_xticks = [i * len(layer_types) + 1 for i in range(len(l2_norm) // len(layer_types))]
        ax.set_xticks(custom_xticks)
        ax.set_xticklabels(range(len(l2_norm) // len(layer_types)))
        ax.set_xlabel('Layer index')
        ax.set_ylabel('L2 norm')
    elif metric == 'l1_norm':
        for layer in layer_types:
            for key, value in l1_norm.items():
                if layer in key:
                    layer_indices[layer].append(index[key])
                    layer_data[layer].append(value)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(l1_norm.values(), '--', c='gray', label='Overall')
        for layer in layer_types:
            ax.plot(layer_indices[layer], layer_data[layer], 's', label=label_mapping[layer])
        ax.plot(index['lm_head'], l1_norm['lm_head'], 's', label='lm_head')
        custom_xticks = [i * len(layer_types) + 1 for i in range(len(l1_norm) // len(layer_types))]
        ax.set_xticks(custom_xticks)
        ax.set_xticklabels(range(len(l1_norm) // len(layer_types)))
        ax.set_xlabel('Layer index')
        ax.set_ylabel('L1 norm')
    ax.legend()
    plt.grid(True)

    plt.tight_layout()
    if title is not None:
        plt.title(title)


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
            label = label
            print(label)
            label = label.replace("per_token", "per_column")
        elif ('W4' not in label and 'W8' not in label) or 'S1' not in label:
            label = label.replace("per_column", "per_token")
        if 'S1' in label:
            print(label)
            label = label.replace("per_token", "per_column")

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
            if 'A4' in label and 'per_token' in label:
                pass
            elif ('W4' not in label and 'W8' not in label) or 'S1' not in label:
                label = label.replace("per_column", "per_token")
            if 'S1' in label:
                print(label)
                label = label.replace("per_token", "per_column")
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
