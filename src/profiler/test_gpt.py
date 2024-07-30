import torch
import argparse
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, GPT2Config
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

try:
    from flash_attn.models.gpt import GPTLMHeadModel
except ImportError:
    from transformers.models.gpt2 import GPT2LMHeadModel as GPTLMHeadModel

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn, RMSNorm
except ImportError:
    layer_norm_fn, RMSNorm = None, None

from pytorch_memory_profiler import forward_benchmark, forward_backward_benchmark, full_benchmark, activation_hook, \
    GB_scale, forward_backward_benchmark_data, forward_benchmark_data, full_benchmark_data


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


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input_ids, position_ids):
        return input_ids


try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None
from collections import OrderedDict, namedtuple


class Warpper(GPTLMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        hidden_states = input_ids
        if self.transformer.parallel_block:
            hidden_states2 = None
        residual = None
        inference_params = None
        num_last_tokens = 0
        mixer_kwargs = (
            {"seqlen": hidden_states.shape[1]}
            if self.transformer.process_group is not None and self.transformer.sequence_parallel
            else {}
        )
        if inference_params is not None:
            mixer_kwargs["inference_params"] = inference_params
        for layer in self.transformer.layers:
            if self.transformer.prenorm:
                if not self.transformer.parallel_block:
                    hidden_states, residual = layer(
                        hidden_states, residual, mixer_kwargs=mixer_kwargs
                    )
                else:
                    hidden_states, hidden_states2, residual = layer(
                        hidden_states, hidden_states2, residual, mixer_kwargs=mixer_kwargs
                    )
            else:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)

        if self.transformer.prenorm:
            if not self.transformer.fused_dropout_add_ln:
                dropped = self.transformer.drop_f(hidden_states)
                if not self.transformer.parallel_block:
                    residual = (dropped + residual) if residual is not None else dropped
                else:
                    dropped2 = self.transformer.drop_f(hidden_states2)
                    residual = (
                        (residual + dropped + dropped2)
                        if residual is not None
                        else dropped + dropped2
                    )
                hidden_states = self.transformer.ln_f(residual.to(dtype=self.transformer.ln_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                hidden_states = layer_norm_fn(
                    hidden_states,
                    self.transformer.ln_f.weight,
                    self.transformer.ln_f.bias,
                    residual=residual,
                    x1=None if not self.transformer.parallel_block else hidden_states2,
                    eps=self.transformer.ln_f.eps,
                    dropout_p=self.transformer.drop_f.p if self.transformer.training else 0.0,
                    prenorm=False,
                    is_rms_norm=isinstance(self.transformer.ln_f, RMSNorm)
                )

        # return hidden_states
        if inference_params is not None:
            assert hidden_states.ndim == 3, "sequence_parallel is not supported in generation mode"
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        lm_logits = self.lm_head(hidden_states)

        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)


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
        model = GPT2LMHeadModel(config).to(device_type)
        prefix_peak_memory_name = os.path.join(args.save_path, f"{args.model}_HF_{args.seq_len}")
    else:
        config.use_flash_attn = True
        config.fused_bias_fc = False
        config.fused_mlp = False
        config.fused_dropout_add_ln = False
        config.residual_in_fp32 = True
        config.pad_vocab_size_multiple = 8
        # model = GPTLMHeadModel(config).to(device_type)
        model = Warpper(config).to(device_type)
        prefix_peak_memory_name = os.path.join(args.save_path, f"{args.model}_{args.seq_len}")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    tokenized_datasets = torch.randint(0, model.config.vocab_size, (1000, args.seq_len))
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.repeats,
    )
    model.train()

    if args.compile:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    # estimated memory:
    batch = torch.randn(args.batch_size, args.seq_len, config.n_embd)
    output_activation = activation_hook(model, batch.to(device_type), device_type=device_type,
                                        amp=args.amp)
    activation_size = sum([v.numel() * v.element_size() for k, v in output_activation.items() if
                           k.startswith('transformer') or k.startswith('lm_head')]) / GB_scale
    logit_size = args.batch_size * args.seq_len * config.vocab_size * 4 / GB_scale
    parameter_size = sum([p.numel() * p.element_size() for p in model.parameters()]) / GB_scale
    states_size = sum([p.numel() * p.element_size() * 2 for p in model.parameters() if p.requires_grad]) / GB_scale
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
        data = torch.randn(args.batch_size, args.seq_len, config.n_embd).requires_grad_(True)
        forward_benchmark_data(model, data, repeats=args.repeats, amp=args.amp, device_type=device_type,
                               categorizes=args.categorizes, prefix_peak_memory_name=prefix_peak_memory_name,
                               verbose=args.verbose)
    elif args.forward_backward:
        data = torch.randn(args.batch_size, args.seq_len, config.n_embd).requires_grad_(True)
        forward_backward_benchmark_data(model, data, repeats=args.repeats, amp=args.amp, device_type=device_type,
                                        categorizes=args.categorizes, prefix_peak_memory_name=prefix_peak_memory_name,
                                        verbose=args.verbose)
    else:
        data = torch.randn(args.batch_size, args.seq_len, config.n_embd).requires_grad_(True)
        label = next(iter(train_dataloader))['labels']
        full_benchmark_data(model, data, label, criterion, optimizer, lr_scheduler, repeats=args.repeats, amp=args.amp,
                            categorizes=args.categorizes, prefix_peak_memory_name=prefix_peak_memory_name,
                            device_type=device_type, verbose=args.verbose)
