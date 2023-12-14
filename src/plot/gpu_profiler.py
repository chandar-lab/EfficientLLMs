import torch
import torch.nn as nn
import gc
import pandas as pd
# modified code from: https://github.com/quentinf00/article-memory-log/tree/master
# use _get_tensors from: https://github.com/li-js/gpu_memory_profiling


def _get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def _get_tensors(gpu_only=True):
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


def _log_tensor(tensor_dict, layer_type=None, hook_type=None, idx=None):
    mem_all, mem_cached = _get_gpu_mem()
    torch.cuda.synchronize()
    for x in _get_tensors():
        if not x in tensor_dict.keys():
            size_byte = {torch.int64: 8, torch.float64: 8, torch.int32: 4, torch.float32: 4, torch.float16: 2,
                         torch.int8: 1}
            estimate_mem = 0
            # for k in tensor_dict.keys():
            #     estimate_mem += tensor_dict[k]['estimate_mem']
            tensor_dict[x] = {"type": type(x), "size": x.numel(),
                              "dtype": x.dtype, "shape": x.shape,
                              "layer_type": layer_type,
                              'hook_type': hook_type,
                              'mem_all': mem_all,
                              'mem_cached': mem_cached,
                              'layer_idx': idx,
                              'estimate_mem': estimate_mem + size_byte[x.dtype] * x.numel()}


def _log_all_tensor(tensor_dict, step, exp_avg_list, exp_avg_sq_list, grad_list):
    mem_all, mem_cached = _get_gpu_mem()
    torch.cuda.synchronize()
    for x in _get_tensors():
        if not x in tensor_dict.keys():
            size_byte = {torch.int64: 8, torch.float64: 8, torch.int32: 4, torch.float32: 4, torch.float16: 2,
                         torch.int8: 1}
            estimate_mem = 0
            for k in tensor_dict.keys():
                estimate_mem += tensor_dict[k]['estimate_mem']
            if isinstance(x, torch.nn.parameter.Parameter):
                data_type = 'param'
            elif id(x) in exp_avg_list or id(x) in exp_avg_sq_list:
                data_type = 'state'
            elif id(x) in grad_list:
                data_type = 'grad'
            else:
                data_type = 'activation'

            tensor_dict[x] = {"step": step, "data_type": data_type, "size": x.numel(),
                              "dtype": x.dtype, "shape": x.shape,
                              'mem_all': mem_all,
                              'mem_cached': mem_cached, "estimate_mem_sum": estimate_mem + (size_byte[x.dtype] * x.numel()),
                              'estimate_mem': size_byte[x.dtype] * x.numel()}


def _generate_mem_hook(handle_ref, idx, hook_type, tensor_dict):
    def hook(self, *args):
        _log_tensor(tensor_dict, layer_type=type(self).__name__,
                    hook_type=hook_type, idx=idx)

    return hook


def _add_memory_hooks(idx, mod, tensor_dict_log, hr):
    h = mod.register_forward_pre_hook(_generate_mem_hook(hr, idx, 'pre', tensor_dict_log))
    hr.append(h)

    h = mod.register_forward_hook(_generate_mem_hook(hr, idx, 'fwd', tensor_dict_log))
    hr.append(h)

    h = mod.register_backward_hook(_generate_mem_hook(hr, idx, 'bwd', tensor_dict_log))
    hr.append(h)


class GPU_Profiler(object):
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim,
                 save_path: str = None, tensor_dict_log: dict = None,
                 log_each_opearation: bool = False):
        if tensor_dict_log is None:
            tensor_dict_log = {}
        self.tensor_dict_log = tensor_dict_log
        self.hr = []
        self.save_path = save_path
        self.step_counter = 0
        self.log_each_opearation = log_each_opearation
        self.optimizer = optimizer
        if self.log_each_opearation:
            for idx, module in enumerate(model.modules()):
                _add_memory_hooks(idx, module, tensor_dict_log, self.hr)

    def start(self):
        if self.log_each_opearation:
            _log_tensor(self.tensor_dict_log, hook_type='init')
        else:
            exp_avg_list = [id(self.optimizer.state[x]['exp_avg']) for x in self.optimizer.state.keys()]
            exp_avg_sq_list = [id(self.optimizer.state[x]['exp_avg_sq']) for x in self.optimizer.state.keys()]
            grad_list = [id(x.grad) for x in self.optimizer.state.keys()]
            _log_all_tensor(self.tensor_dict_log, -1, exp_avg_list, exp_avg_sq_list, grad_list)

    def stop(self):
        [h.remove() for h in self.hr]

    def save_result(self):
        df = pd.DataFrame(self.tensor_dict_log.values())
        if self.save_path is not None:
            df.to_csv(self.save_path)
        idx = df.step < 1
        print(f"sum memory: {df[idx].estimate_mem.sum()}")
        print(df)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        self.save_result()

    def step(self):
        if not self.log_each_opearation:
            exp_avg_list = [id(self.optimizer.state[x]['exp_avg']) for x in self.optimizer.state.keys()]
            exp_avg_sq_list = [id(self.optimizer.state[x]['exp_avg_sq']) for x in self.optimizer.state.keys()]
            grad_list = [id(x.grad) for x in self.optimizer.state.keys()]
            _log_all_tensor(self.tensor_dict_log, self.step_counter, exp_avg_list, exp_avg_sq_list, grad_list)
        self.step_counter += 1


if __name__ == "__main__":
    assert torch.cuda.is_available()

    # gpt2 example
    # import torch
    # from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
    # from torch.utils.data import DataLoader
    # from transformers import AdamW, get_scheduler
    #
    # context_length = 1024
    # num_training_steps = 5
    # batch_size = 16
    # device = torch.device('cuda')
    #
    # model_name = "gpt2"
    # model = GPT2LMHeadModel.from_pretrained(model_name)
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # tokenized_datasets = torch.randint(0, model.config.vocab_size, (1000, context_length))
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    #
    # optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps,
    # )
    # model = model.to(device)
    # model.train()
    #
    # with GPU_Profiler(model, optimizer=optimizer, save_path='tensor_dict_log.csv') as profiler:
    #     for step in range(num_training_steps):
    #         batch = next(iter(train_dataloader))
    #         batch = {k: v.to(model.device) for k, v in batch.items()}
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         lr_scheduler.step()
    #         profiler.step()

    # resnet18 example
    # import torch
    # from torch.optim import AdamW
    # import torch.optim.lr_scheduler as lr_scheduler
    # from torchvision.models import resnet18
    #
    # batch_size = 16
    # num_training_steps = 5
    # device = torch.device('cuda')
    # model = resnet18()
    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    # model = model.to(device)
    # model.train()
    #
    # with GPU_Profiler(model, optimizer=optimizer, save_path='tensor_dict_log.csv') as profiler:
    #     for step in range(num_training_steps):
    #         x = torch.randn(batch_size, 3, 224, 224).to(device)
    #         label = torch.randint(0, 1000, (batch_size,)).to(device)
    #         output = model(x)
    #         loss = criterion(output, label)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #         profiler.step()

    # toy example
    import torch
    from torch.optim import AdamW
    import torch.optim.lr_scheduler as lr_scheduler

    batch_size = 16
    num_training_steps = 3
    device = torch.device('cuda')
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10),
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    model = model.to(device)
    model.train()

    with GPU_Profiler(model, optimizer=optimizer, save_path='tensor_dict_log.csv') as profiler:
        for step in range(num_training_steps):
            x = torch.randn(batch_size, 100).to(device)
            label = torch.randint(0, 10, (batch_size,)).to(device)
            output = model(x)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            profiler.step()





