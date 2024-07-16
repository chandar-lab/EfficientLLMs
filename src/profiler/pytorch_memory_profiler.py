# modify code from: https://pytorch.org/blog/understanding-gpu-memory-1/?utm_content=275432243&utm_medium=social&utm_source=twitter&hss_channel=tw-776585502606721024

import os
import logging
import socket
import torch
import pandas as pd
from datetime import datetime
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline
from torch.profiler._memory_profiler import _CATEGORY_TO_COLORS, _CATEGORY_TO_INDEX
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from torch.autograd.profiler import record_function

MB_scale = 1024 * 1024
GB_scale = 1024 * 1024 * 1024
# Keep a max of 100,000 alloc/free events in the recorded history leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def rand_like(input: torch.tensor):
    return torch.randn_like(input.float()).to(input.dtype)


def set_plot_style(
        fsize: int = 14,
        tsize: int = 14,
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


def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )


def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return


def forward_benchmark(model,
                      train_dataloader,
                      repeats=10,
                      amp=True,
                      amp_dtype=torch.bfloat16,
                      categorizes=False,
                      verbose=False,
                      prefix_peak_memory_name="PEAK_MEMORY_FORWARD.csv",
                      device_type='cuda', ):
    def amp_wrapper(model, input):
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp):
            output = model(input)

    if device_type == "cpu":
        activities = [torch.profiler.ProfilerActivity.CPU, ]
    else:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA, ]

    if categorizes:
        with torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=repeats // 2, active=repeats // 2, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                on_trace_ready=trace_handler,
        ) as prof:
            for i, batch in enumerate(train_dataloader):
                if i >= repeats:
                    break
                prof.step()
                prof.prefix_peak_memory_name = prefix_peak_memory_name
                if device_type == 'cuda':
                    prof.device_type = 'cuda:0'
                else:
                    prof.device_type = device_type
                if isinstance(batch, list):
                    prof.batch_size = batch[0].shape[0]
                    amp_wrapper(model, batch[0].to(device_type))
                else:
                    batch = {k: v.to(device_type) for k, v in batch.items()}
                    prof.batch_size = batch['input_ids'].shape[0]
                    amp_wrapper(model, batch['input_ids'])
            if verbose:
                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    else:
        # Start recording memory snapshot history
        start_record_memory_history()
        for i, batch in enumerate(train_dataloader):
            if i >= repeats:
                break
            if isinstance(batch, list):
                amp_wrapper(model, batch[0].to(device_type))
            else:
                batch = {k: v.to(device_type) for k, v in batch.items()}
                amp_wrapper(model, batch['input_ids'])
        # Create the memory snapshot file
        export_memory_snapshot()
        # Stop recording memory snapshot history
        stop_record_memory_history()


def forward_benchmark_data(model,
                           input_data,
                           repeats=10,
                           amp=True,
                           amp_dtype=torch.bfloat16,
                           categorizes=False,
                           verbose=False,
                           prefix_peak_memory_name="PEAK_MEMORY_FORWARD.csv",
                           device_type='cuda', ):
    def amp_wrapper(model, input):
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp):
            output = model(input)

    if device_type == "cpu":
        activities = [torch.profiler.ProfilerActivity.CPU, ]
    else:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA, ]

    if categorizes:
        with torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=repeats // 2, active=repeats // 2, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                on_trace_ready=trace_handler,
        ) as prof:
            for i in range(repeats):
                prof.step()
                prof.prefix_peak_memory_name = prefix_peak_memory_name
                if device_type == 'cuda':
                    prof.device_type = 'cuda:0'
                else:
                    prof.device_type = device_type
                prof.batch_size = input_data.shape[0]
                amp_wrapper(model, input_data.to(device_type))
            if verbose:
                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    else:
        raise 'kir hast felan'


def forward_backward_benchmark(
        model,
        train_dataloader,
        repeats=10,
        amp=True,
        amp_dtype=torch.bfloat16,
        categorizes=False,
        verbose=False,
        prefix_peak_memory_name="PEAK_MEMORY_FORWARD_BACKWARD.csv",
        device_type='cuda',
):
    def amp_wrapper(model, input):
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp):
            output = model(input)
            if type(output) is tuple:
                output = output[0]
            grad = torch.randn_like(output)
            output.backward(grad, retain_graph=True)

    if device_type == "cpu":
        activities = [torch.profiler.ProfilerActivity.CPU, ]
    else:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA, ]

    if categorizes:
        with torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=repeats // 2, active=repeats // 2, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                on_trace_ready=trace_handler,
        ) as prof:
            for i, batch in enumerate(train_dataloader):
                if i >= repeats:
                    break
                prof.step()
                prof.prefix_peak_memory_name = prefix_peak_memory_name
                if device_type == 'cuda':
                    prof.device_type = 'cuda:0'
                else:
                    prof.device_type = device_type
                if isinstance(batch, list):
                    prof.batch_size = batch[0].shape[0]
                    x = batch[0].to(device_type).requires_grad_(True)
                    amp_wrapper(model, x)
                else:
                    batch = {k: v.to(device_type) for k, v in batch.items()}
                    prof.batch_size = batch['input_ids'].shape[0]
                    x = batch['input_ids'].requires_grad_(True)
                    amp_wrapper(model, x)
            if verbose:
                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    else:
        # Start recording memory snapshot history
        start_record_memory_history()
        for i, batch in enumerate(train_dataloader):
            if i >= repeats:
                break
            if isinstance(batch, list):
                x = batch[0].to(device_type).requires_grad_(True)
                amp_wrapper(model, x)
            else:
                batch = {k: v.to(device_type) for k, v in batch.items()}
                x = batch['input_ids'].requires_grad_(True)
                amp_wrapper(model, x)
        # Create the memory snapshot file
        export_memory_snapshot()
        # Stop recording memory snapshot history
        stop_record_memory_history()


def forward_backward_benchmark_data(
        model,
        input_data,
        repeats=10,
        amp=True,
        amp_dtype=torch.bfloat16,
        categorizes=False,
        verbose=False,
        prefix_peak_memory_name="PEAK_MEMORY_FORWARD_BACKWARD.csv",
        device_type='cuda',
):
    def amp_wrapper(model, input):
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp):
            output = model(input)
            if isinstance(output, CausalLMOutputWithCrossAttentions):
                output = output.logits
            if type(output) is tuple:
                output = output[0]
            grad = torch.randn_like(output)
            output.backward(grad, retain_graph=True)

    if device_type == "cpu":
        activities = [torch.profiler.ProfilerActivity.CPU, ]
    else:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA, ]

    if categorizes:
        with torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=repeats // 2, active=repeats // 2, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                on_trace_ready=trace_handler,
        ) as prof:
            for i in range(repeats):
                prof.step()
                prof.prefix_peak_memory_name = prefix_peak_memory_name
                if device_type == 'cuda':
                    prof.device_type = 'cuda:0'
                else:
                    prof.device_type = device_type

                prof.batch_size = input_data.shape[0]
                amp_wrapper(model, input_data.to(device_type))
            if verbose:
                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    else:
        raise 'kir hast felan'


def full_benchmark(
        model,
        train_dataloader,
        criterion,
        optimizer,
        scheduler,
        repeats=10,
        amp=True,
        amp_dtype=torch.bfloat16,
        categorizes=False,
        verbose=False,
        prefix_peak_memory_name="PEAK_MEMORY.csv",
        device_type='cuda',
):
    def amp_wrapper(model, input, label, criterion, optimizer, scheduler):
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp):
            output = model(input)
            if hasattr(output, 'logits'):
                output = output.logits
            loss = criterion(output.view(-1, output.size(-1)), label.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    if device_type == "cpu":
        activities = [torch.profiler.ProfilerActivity.CPU, ]
    else:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA, ]

    model.train()
    if categorizes:
        with torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=repeats // 2, active=repeats // 2, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                on_trace_ready=trace_handler,
        ) as prof:
            for i, batch in enumerate(train_dataloader):
                if i >= repeats:
                    break
                prof.step()
                prof.prefix_peak_memory_name = prefix_peak_memory_name
                if device_type == 'cuda':
                    prof.device_type = 'cuda:0'
                else:
                    prof.device_type = device_type
                if isinstance(batch, list):
                    prof.batch_size = batch[0].shape[0]
                    amp_wrapper(model, batch[0].to(device_type), batch[1].to(device_type), criterion, optimizer,
                                scheduler, )
                else:
                    batch = {k: v.to(device_type) for k, v in batch.items()}
                    prof.batch_size = batch['input_ids'].shape[0]
                    amp_wrapper(model, batch['input_ids'], batch['labels'], criterion, optimizer, scheduler)
            if verbose:
                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    else:
        # Start recording memory snapshot history
        start_record_memory_history()
        for i, batch in enumerate(train_dataloader):
            if i >= repeats:
                break
            if isinstance(batch, list):
                amp_wrapper(model, batch[0].to(device_type), batch[1].to(device_type), criterion, optimizer, scheduler)
            else:
                batch = {k: v.to(device_type) for k, v in batch.items()}
                amp_wrapper(model, batch['input_ids'], batch['labels'], criterion, optimizer, scheduler)
        # Create the memory snapshot file
        export_memory_snapshot()
        # Stop recording memory snapshot history
        stop_record_memory_history()


def full_benchmark_data(
        model,
        input_data,
        input_label,
        criterion,
        optimizer,
        scheduler,
        repeats=10,
        amp=True,
        amp_dtype=torch.bfloat16,
        categorizes=False,
        verbose=False,
        prefix_peak_memory_name="PEAK_MEMORY.csv",
        device_type='cuda',
):
    def amp_wrapper(model, input, label, criterion, optimizer, scheduler):
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp):
            output = model(input)
            if hasattr(output, 'logits'):
                output = output.logits
            loss = criterion(output.view(-1, output.size(-1)), label.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    if device_type == "cpu":
        activities = [torch.profiler.ProfilerActivity.CPU, ]
    else:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA, ]

    model.train()
    if categorizes:
        with torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=repeats // 2, active=repeats // 2, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                on_trace_ready=trace_handler,
        ) as prof:
            for i in range(repeats):
                prof.step()
                prof.prefix_peak_memory_name = prefix_peak_memory_name
                if device_type == 'cuda':
                    prof.device_type = 'cuda:0'
                else:
                    prof.device_type = device_type

                prof.batch_size = input_data.shape[0]
                amp_wrapper(model, input_data.to(device_type), input_label.to(device_type), criterion, optimizer,
                            scheduler, )

            if verbose:
                print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    else:
        # Start recording memory snapshot history
        raise NotImplementedError


def export_memory_timeline(prof, path, figsize=(20, 12), title=None):
    mem_res = MemoryProfile(prof.profiler.kineto_results)
    mem_tl = MemoryProfileTimeline(mem_res)
    mt = mem_tl._coalesce_timeline(device_str=prof.device_type)
    times, sizes = np.array(mt[0]), np.array(mt[1])
    t_min = min(times)
    times -= t_min
    stacked = np.cumsum(sizes, axis=1) / 1024 ** 3
    np.save(f"{prof.prefix_peak_memory_name}_stacked_memory.npy", stacked)
    np.save(f"{prof.prefix_peak_memory_name}_stacked_time.npy", times)

    device = torch.device(prof.device_type)
    max_memory_allocated = torch.cuda.max_memory_allocated(device)
    max_memory_reserved = torch.cuda.max_memory_reserved(device)

    # Calculate peak memory time and percentages
    peak_memory_index = stacked[:, -1].argmax()
    peak_memory_time = times[peak_memory_index]
    peak_memory_values = stacked[peak_memory_index]
    memory_result = {}
    memory_usage = {}
    total_peak_memory = peak_memory_values[-1]
    logger.info(
        f'mem_all: {total_peak_memory}, max_memory_allocated: {max_memory_allocated / GB_scale}, max_memory_reserved: {max_memory_reserved / GB_scale}')

    for category, color in _CATEGORY_TO_COLORS.items():
        i = _CATEGORY_TO_INDEX[category]
        mem_usage = peak_memory_values[i + 1] - peak_memory_values[i]
        memory_result[category.name if category else 'OTHER'] = round(mem_usage / total_peak_memory * 100, 3)
        memory_usage[category.name if category else 'OTHER'] = mem_usage

    for k, v in memory_usage.items():
        logger.info(f'{k}: {v}')

    # save peak memory result
    save_path = f"{prof.prefix_peak_memory_name}_PEAK_MEMORY.csv"
    if os.path.exists(save_path):
        all_memory_result = pd.read_csv(save_path, index_col=0)
    else:
        all_memory_result = pd.DataFrame()
    df_new_row = pd.DataFrame(memory_usage, index=[prof.batch_size])
    all_memory_result = pd.concat([all_memory_result, df_new_row])
    all_memory_result.to_csv(save_path)

    # Plot memory timeline as stacked data
    set_plot_style()
    fig = plt.figure(figsize=figsize)
    axes = fig.gca()
    for category, color in _CATEGORY_TO_COLORS.items():
        i = _CATEGORY_TO_INDEX[category]
        axes.fill_between(
            times / 1e6, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7
        )

    # Add a vertical dashed line at the peak memory time
    axes.axvline(x=peak_memory_time / 1e6, color='red', linestyle='--', linewidth=2.5)

    # Annotate the percentages below the peak memory line, aligned to the left
    y_pos = -0.1 * max_memory_reserved / (10 ** 9)
    annotation_text = "\n".join([f"{k}: {v:.1f}%" for k, v in memory_result.items()])
    text_box = axes.text(peak_memory_time / 1e6, y_pos, annotation_text, color='black', ha='left', va='top',
                         bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    fig.legend(["Unknown" if i is None else i.name for i in _CATEGORY_TO_COLORS])
    axes.set_xlabel("Time (ms)")
    axes.set_ylabel("Memory (GB)")
    title = "\n\n".join(
        ([title] if title else [])
        + [
            f"{prof.prefix_peak_memory_name.split('/')[-1]} \n"
            f"Max memory allocated: {max_memory_allocated / (10 ** 9):.2f} GB \n"
            f"Max memory reserved: {max_memory_reserved / (10 ** 9):.2f} GB"
        ]
    )
    axes.set_title(title)
    fig.savefig(path, dpi=300, pad_inches=.1, bbox_inches='tight')
    logger.info(f'save results in {prof.prefix_peak_memory_name}')


def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{prof.prefix_peak_memory_name}_{timestamp}"

    # # Construct the trace file.
    # prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    # prof.export_memory_timeline(f"{file_prefix}.html", device=prof.device_type)
    export_memory_timeline(prof, path=f"{file_prefix}.png")


from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions


def activation_hook(model, inputs, device_type='cuda', amp=True, amp_dtype=torch.bfloat16):
    # add hook to record the min max value of the activation
    output_activation = {}

    def add_range_recoder_hook(model):
        import functools
        def _record_range(self, x, y, module_name):
            if isinstance(x, tuple):
                x = x[0]
            if isinstance(y, tuple):
                y = y[0]
            elif isinstance(y, BaseModelOutputWithPastAndCrossAttentions):
                y = y.last_hidden_state
            elif isinstance(y, CausalLMOutputWithCrossAttentions):
                y = y.logits
            output_activation[module_name] = y.detach()

        all_hooks = []
        for name, m in model.named_modules():
            all_hooks.append(m.register_forward_hook(
                functools.partial(_record_range, module_name=name)))
        return all_hooks

    hooks = add_range_recoder_hook(model)
    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp):
        model(inputs)

    # remove hooks
    for h in hooks:
        h.remove()
    return output_activation
