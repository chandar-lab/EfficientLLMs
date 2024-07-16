import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import os
import torch
from transformers import AdamW
from torch.utils.data import Dataset, DataLoader
from pytorch_memory_profiler import forward_benchmark, forward_backward_benchmark, full_benchmark, activation_hook, GB_scale

def args_parser():
    parser = argparse.ArgumentParser(
        description='profile time',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--seq_len', default=64, type=int, help='sequence length')
    parser.add_argument('--repeats', default=10, type=int, help='number of repeat in profiling')
    parser.add_argument('--model', default='mlp', type=str, help='name of model')
    parser.add_argument('--save_path', default='../profile_results', type=str, help='save path')
    parser.add_argument('--amp', dest='amp', action='store_true', help='use AMP')
    parser.add_argument('--forward', action='store_true', help='profile for forward only')
    parser.add_argument('--forward_backward', action='store_true', help='profile for forward and backward')
    parser.add_argument('--categorizes', dest='categorizes', action='store_true',
                        help='use categorizes profiling to indication param, activation, grad, state, ...')
    parser.add_argument('--compile', action='store_true', help='use torch.compile and TF cores')
    parser.add_argument('--verbose', action='store_true', help='show profile time and memory results')
    args = parser.parse_args()
    return args


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers):
        super(SimpleMLP, self).__init__()
        self.num_layers = num_layers
        modules = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        modules.append(nn.Linear(hidden_dim, num_classes))
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers):
        super(ResidualMLP, self).__init__()
        self.num_layers = num_layers
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.residual_blocks.append(block)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.input_layer(x)
        for block in self.residual_blocks:
            residual = out
            out = block(out)
            out = out + residual
            out = nn.ReLU()(out)
        out = self.output_layer(out)
        return out


class ToyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":

    args = args_parser()
    os.makedirs(args.save_path, exist_ok=True)
    if torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'

    data = torch.randn(1000, args.seq_len, 32)
    labels = torch.randint(0, 10, (1000, args.seq_len))
    dataset = ToyDataset(data, labels)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.model == 'mlp':
        prefix_peak_memory_name = os.path.join(args.save_path, f"ToyModel")
        model = SimpleMLP(input_dim=32, hidden_dim=512, num_classes=10, num_layers=16).to(device_type)
    elif args.model == 'resmlp':
        prefix_peak_memory_name = os.path.join(args.save_path, f"ToyModel")
        model = ResidualMLP(input_dim=32, hidden_dim=512, num_classes=10, num_layers=16).to(device_type)
    else:
        raise NotImplementedError
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

    if args.compile:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    # estimated memory:
    batch = next(iter(train_dataloader))
    output_activation = activation_hook(model, batch[0].to(device_type), device_type=device_type, amp=args.amp)
    activation_size = sum([v.numel() * v.element_size() for k,v in output_activation.items()])/GB_scale
    parameter_size = sum([p.numel() * p.element_size() for p in model.parameters()]) / GB_scale
    states_size = sum([p.numel() * p.element_size() * 2 for p in model.parameters() if p.requires_grad]) / GB_scale
    print(
        f"========= estimated memory usage =========\n"
        f"PARAMETER:: {parameter_size},\n"
        f"OPTIMIZER_STATE:: {states_size},\n"
        f"ACTIVATION:: {activation_size},\n"
        f"TOTAL: {activation_size + parameter_size + states_size }\n"
        f"==========================================="
    )

    if args.forward:
        forward_benchmark(model, train_dataloader, repeats=args.repeats, amp=args.amp, device_type=device_type,
                          categorizes=args.categorizes, prefix_peak_memory_name=prefix_peak_memory_name, verbose=args.verbose)
    elif args.forward_backward:
        forward_backward_benchmark(model, train_dataloader, repeats=args.repeats, amp=args.amp, device_type=device_type,
                                   categorizes=args.categorizes, prefix_peak_memory_name=prefix_peak_memory_name, verbose=args.verbose)
    else:
        full_benchmark(model, train_dataloader, criterion, optimizer, scheduler, repeats=args.repeats, amp=args.amp,
                       categorizes=args.categorizes, prefix_peak_memory_name=prefix_peak_memory_name, device_type=device_type, verbose=args.verbose)
