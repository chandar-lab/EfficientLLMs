import torch
import torch.nn as nn
import gc
import pandas as pd
import argparse
import GPUtil as GPU
from flash_attn.models.gpt import GPTLMHeadModel
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import GPT2Config
from torch.nn import CrossEntropyLoss
from pynvml import *

# modified code from: https://github.com/quentinf00/article-memory-log/tree/master
# use _get_tensors from: https://github.com/li-js/gpu_memory_profiling

MB_scale = 1024 * 1024
GB_scale = 1024 * 1024 * 1024
size_byte = {torch.int64: 8, torch.float64: 8, torch.int32: 4, torch.float32: 4,
             torch.float16: 2, torch.bfloat16: 2, torch.int8: 1, torch.bool: 0.125, torch.uint8: 1}


def args_parser():
    parser = argparse.ArgumentParser(
        description='profile time',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--seq_len', default=1024, type=int, help='sequence length')
    parser.add_argument('--repeats', default=5, type=int, help='number of repeat in profiling')
    parser.add_argument('--save_path', default='../save', type=str, help='save path')
    parser.add_argument('--amp', dest='amp', action='store_true', help='use AMP')
    parser.add_argument('--model', default='gpt2-small', type=str, help='name of model')
    parser.add_argument('--hf_model', action='store_true', help='use HuggingFace model')
    parser.add_argument('--forward', dest='forward', action='store_true', help='profile for foward only')
    parser.add_argument('--log_each_opearation', dest='log_each_opearation', action='store_true',
                        help='log for each operation')
    parser.add_argument('--gpu_only', dest='gpu_only', action='store_true',
                        help='log tensors only on gpu')
    args = parser.parse_args()
    return args


def _update_dict(dict, new_dict):
    for key, value in new_dict.items():
        if key in dict:
            dict[key].append(value)
        else:
            dict[key] = [value]


def _get_gpu_mem(synchronize=True, empty_cache=True):
    if empty_cache:
        gc.collect()
        torch.cuda.empty_cache()
    if synchronize:
        torch.cuda.synchronize()
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def _get_gpu_mem_nvidia(synchronize=True, empty_cache=True):
    # if empty_cache:
    #     gc.collect()
    #     torch.cuda.empty_cache()
    # if synchronize:
    #     torch.cuda.synchronize()
    GPUs = GPU.getGPUs()
    gpu = GPUs[0]
    return gpu.memoryUsed
    # nvmlInit()
    # handle = nvmlDeviceGetHandleByIndex(0)
    # info = nvmlDeviceGetMemoryInfo(handle)
    # return info.used/MB_scale


def _get_tensors(gpu_only=False):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if gpu_only:
                if tensor.is_cuda:
                    yield tensor
            else:
                yield tensor
        except Exception as e:
            pass


def _log_tensor(tensor_dict, layer_type=None, hook_type=None, idx=None, exp_avg_list=[], exp_avg_sq_list=[],
                grad_list=[], gpu_only=False):
    if gpu_only:
        mem_all, mem_cached = _get_gpu_mem()
        mem_nvidia = _get_gpu_mem_nvidia()
    # tensor_dict = {}
    for x in _get_tensors(gpu_only=gpu_only):
        if 'id' not in tensor_dict.keys() or id(x) not in tensor_dict['id']:
            if isinstance(x, torch.nn.parameter.Parameter):
                data_type = 'param'
            elif id(x) in exp_avg_list or id(x) in exp_avg_sq_list:
                data_type = 'state'
            elif id(x) in grad_list:
                data_type = 'grad'
            else:
                data_type = 'activation'
            if 'estimate_mem' in tensor_dict.keys():
                estimate_mem = sum(tensor_dict['estimate_mem'])
            else:
                estimate_mem = 0
            tensor_dict_ = {"type": type(x), "size": x.numel(),
                            "dtype": x.dtype, "shape": x.shape,
                            'layer_idx': idx,
                            'id': id(x),
                            "layer_type": layer_type,
                            'hook_type': hook_type,
                            'data_type': data_type,
                            "estimate_mem": (size_byte[x.dtype] * x.numel()) / MB_scale,
                            "estimate_mem_sum": (estimate_mem + (size_byte[x.dtype] * x.numel()) / MB_scale)}
            if gpu_only:
                tensor_dict_['mem_all'] = mem_all / MB_scale
                tensor_dict_['mem_cached'] = mem_cached / MB_scale
                tensor_dict_['mem_nvidia'] = mem_nvidia
            _update_dict(tensor_dict, tensor_dict_)


def _log_all_tensor(tensor_dict, step, exp_avg_list, exp_avg_sq_list, grad_list, gpu_only=False):
    if gpu_only:
        mem_all, mem_cached = _get_gpu_mem()
        mem_nvidia = _get_gpu_mem_nvidia()
    # tensor_dict = {}
    for x in _get_tensors(gpu_only):
        if 'id' not in tensor_dict.keys() or id(x) not in tensor_dict['id']:
            if isinstance(x, torch.nn.parameter.Parameter):
                data_type = 'param'
            elif id(x) in exp_avg_list or id(x) in exp_avg_sq_list:
                data_type = 'state'
            elif id(x) in grad_list:
                data_type = 'grad'
            else:
                # if x.requires_grad:
                #     data_type = 'activation'
                # else:
                #     data_type = 'input'
                data_type = 'activation'
            if 'estimate_mem' in tensor_dict.keys():
                estimate_mem = sum(tensor_dict['estimate_mem'])
            else:
                estimate_mem = 0
            tensor_dict_ = {"step": step, "data_type": data_type, "size": x.numel(),
                            "dtype": x.dtype, "shape": x.shape,
                            "id": id(x),
                            "estimate_mem": (size_byte[x.dtype] * x.numel()) / MB_scale,
                            "estimate_mem_sum": (estimate_mem + (size_byte[x.dtype] * x.numel()) / MB_scale)}
            if gpu_only:
                tensor_dict_['mem_all'] = mem_all / MB_scale
                tensor_dict_['mem_cached'] = mem_cached / MB_scale
                tensor_dict_['mem_nvidia'] = mem_nvidia
            _update_dict(tensor_dict, tensor_dict_)

    return tensor_dict


def _generate_mem_hook(handle_ref, idx, hook_type, tensor_dict, exp_avg_list, exp_avg_sq_list, grad_list, gpu_only):
    def hook(self, *args):
        _log_tensor(tensor_dict, layer_type=type(self).__name__,
                    hook_type=hook_type, idx=idx, exp_avg_list=exp_avg_list,
                    exp_avg_sq_list=exp_avg_sq_list, grad_list=grad_list, gpu_only=gpu_only)

    return hook


def _add_memory_hooks(idx, mod, tensor_dict_log, hr, exp_avg_list, exp_avg_sq_list, grad_list, gpu_only):
    h = mod.register_forward_pre_hook(
        _generate_mem_hook(hr, idx, 'pre', tensor_dict_log, exp_avg_list, exp_avg_sq_list, grad_list, gpu_only))
    hr.append(h)

    h = mod.register_forward_hook(
        _generate_mem_hook(hr, idx, 'fwd', tensor_dict_log, exp_avg_list, exp_avg_sq_list, grad_list, gpu_only))
    hr.append(h)

    h = mod.register_full_backward_hook(
        _generate_mem_hook(hr, idx, 'bwd', tensor_dict_log, exp_avg_list, exp_avg_sq_list, grad_list, gpu_only))
    hr.append(h)


class GPU_Profiler(object):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim,
                 save_path: str = None, tensor_dict_log: dict = None,
                 log_each_opearation: bool = False, gpu_only: bool = False):
        if tensor_dict_log is None:
            tensor_dict_log = {}
        self.tensor_dict_log = tensor_dict_log
        self.hr = []
        self.save_path = save_path
        self.step_counter = 0
        self.log_each_opearation = log_each_opearation
        self.optimizer = optimizer
        self.params = list(model.parameters())
        self.gpu_only = gpu_only
        if self.log_each_opearation:
            exp_avg_list, exp_avg_sq_list, grad_list = self.update_grad_states()
            for idx, module in enumerate(model.modules()):
                _add_memory_hooks(idx, module, tensor_dict_log, self.hr, exp_avg_list, exp_avg_sq_list, grad_list,
                                  gpu_only)

    def start(self):
        if self.log_each_opearation:
            _log_tensor(self.tensor_dict_log, hook_type='init', gpu_only=self.gpu_only)
        else:
            exp_avg_list, exp_avg_sq_list, grad_list = self.update_grad_states()
            _log_all_tensor(self.tensor_dict_log, -1, exp_avg_list, exp_avg_sq_list, grad_list, gpu_only=self.gpu_only)

    def stop(self):
        [h.remove() for h in self.hr]

    def save_result(self):
        df = pd.DataFrame(self.tensor_dict_log)
        if self.save_path is not None:
            df.to_csv(self.save_path, index=False)
        print(df)

    def update_grad_states(self):
        grad_list = [id(p.grad) for p in self.params]
        if self.optimizer is not None:
            exp_avg_list = [id(self.optimizer.state[x]['exp_avg']) for x in self.optimizer.state.keys()]
            exp_avg_sq_list = [id(self.optimizer.state[x]['exp_avg_sq']) for x in self.optimizer.state.keys()]
        else:
            exp_avg_list = []
            exp_avg_sq_list = []
        return exp_avg_list, exp_avg_sq_list, grad_list

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        self.save_result()

    def step(self):
        if self.log_each_opearation:
            pass
        else:
            exp_avg_list, exp_avg_sq_list, grad_list = self.update_grad_states()
            _log_all_tensor(self.tensor_dict_log, self.step_counter, exp_avg_list, exp_avg_sq_list, grad_list,
                            gpu_only=self.gpu_only)
        self.step_counter += 1


def flash_atten_forward(x, labels, model):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        lm_logits = model(x).logits
        labels = labels.to(lm_logits.device)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss


if __name__ == "__main__":
    # assert torch.cuda.is_available()

    # # gpt2 example
    # import torch
    # from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
    # from torch.utils.data import DataLoader
    # from transformers import AdamW, get_scheduler

    # args = args_parser()
    # if args.gpu_only:
    #     device = torch.device('cuda')
    #     device_type = 'cuda'
    # else:
    #     device = torch.device('cpu')
    #     device_type = 'cpu'

    # if args.model == 'gpt2-small':
    #     config = GPT2Config()
    # elif args.model == 'gpt2-medium':
    #     config = GPT2Config(n_embd=1024, n_head=16, n_layer=24)
    # elif args.model == 'gpt2-large':
    #     config = GPT2Config(n_embd=1280, n_head=20, n_layer=36)
    # else:
    #     raise 'not provide'
    # if args.hf_model:
    #     model = GPT2LMHeadModel(config).to(device)
    # else:
    #     config.use_flash_attn = True
    #     config.fused_bias_fc = True
    #     config.fused_mlp = True
    #     config.fused_dropout_add_ln = True
    #     config.residual_in_fp32 = True
    #     config.pad_vocab_size_multiple = 8
    #     model = GPTLMHeadModel(config).cuda()

    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # tokenized_datasets = torch.randint(0, model.config.vocab_size, (10000, args.seq_len))
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)

    # optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=args.repeats,
    # )
    # model.train()

    # with GPU_Profiler(model, optimizer=optimizer, save_path='tensor_dict_log.csv',
    #                   log_each_opearation=args.log_each_opearation, gpu_only=args.gpu_only) as profiler:
    #     for step, batch in enumerate(train_dataloader):
    #         if step >= args.repeats:
    #             break
    #     #for step in range(args.repeats):
    #         batch = next(iter(train_dataloader))
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         # outputs = model(**batch)
    #         # loss = outputs.loss
    #         loss = flash_atten_forward(batch['input_ids'], batch['labels'], model)
    #         # optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         lr_scheduler.step()
    #         profiler.step()

    # # resnet18 example
    # import torch
    # from torch.optim import AdamW
    # import torch.optim.lr_scheduler as lr_scheduler
    # from torchvision.models import resnet18

    # args = args_parser()
    # if args.gpu_only:
    #     device = torch.device('cuda')
    #     device_type = 'cuda'
    # else:
    #     device = torch.device('cpu')
    #     device_type = 'cpu'
    # model = resnet18().to(device)
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    # model.train()

    # with GPU_Profiler(model, optimizer=optimizer, save_path='tensor_dict_log.csv',
    #                   log_each_opearation=args.log_each_opearation, gpu_only=args.gpu_only) as profiler:
    #     for step in range(args.repeats):
    #         x = torch.randn(args.batch_size, 3, 224, 224).to(device)
    #         label = torch.randint(0, 1000, (args.batch_size,)).to(device)
    #         output = model(x)
    #         loss = criterion(output, label)
    #         # optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #         profiler.step()

    # toy example
    import torch
    from torch.optim import AdamW
    import torch.optim.lr_scheduler as lr_scheduler

    args = args_parser()
    if args.gpu_only:
        device = torch.device('cuda')
        device_type = 'cuda'
    else:
        device = torch.device('cpu')
        device_type = 'cpu'

    model = nn.Sequential(
        nn.Linear(1024, 4096),
        # nn.ReLU(),
        # nn.Linear(4096, 4096),
        # nn.ReLU(),
        # nn.Linear(4096, 4096),
        # nn.ReLU(),
        # nn.Linear(4096, 4096),
        # nn.ReLU(),
        # nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 10),
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    model.train()
    with GPU_Profiler(model, optimizer=optimizer, save_path='tensor_dict_log.csv',
                      log_each_opearation=args.log_each_opearation, gpu_only=args.gpu_only) as profiler:
        for step in range(args.repeats):
            x = torch.randn(args.batch_size, 1024).to(device)
            label = torch.randint(0, 10, (args.batch_size,)).to(device)
            output = model(x)
            loss = criterion(output, label)
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            profiler.step()

    # # linear layer example
    # import torch
    # from torch.optim import AdamW
    # import torch.optim.lr_scheduler as lr_scheduler
    #
    # args = args_parser()
    # if args.gpu_only:
    #     device = torch.device('cuda')
    #     device_type = 'cuda'
    # else:
    #     device = torch.device('cpu')
    #     device_type = 'cpu'
    # m_linear = torch.nn.Linear(768, 2304).to(device)
    # x = torch.randn(args.batch_size, args.seq_len, 768).to(device).requires_grad_(True)
    #
    # with GPU_Profiler(m_linear, optimizer=None, save_path='tensor_dict_log.csv',
    #                   log_each_opearation=args.log_each_opearation, gpu_only=args.gpu_only) as profiler:
    #     for step in range(args.repeats):
    #         y = m_linear(x)
    #         grad = torch.randn_like(y)
    #         y.backward(grad, retain_graph=True)
    #         profiler.step()
