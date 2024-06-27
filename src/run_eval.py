import argparse
import json
import numpy as np
import random
import os

from lm_eval import tasks
import copy
from runtime import Runtime
from common import Params, load_configs, creat_unique_experiment_name

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', required=True, default='configs/base.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp.jsonnet')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 1234, 1337, 2024, 1357],
                        help='list of seed integers for random number generation')
    parser.add_argument('--tasks', default="GLUE")
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    _config = load_configs(args.configs)
    exp_name = creat_unique_experiment_name(_config)
    _config_copy = copy.deepcopy(_config)
    experiment = Runtime.from_params(Params(_config), exp_name=exp_name, _config_copy=_config_copy)

    results = {}
    if args.tasks == 'GLUE':
        task_names = ['mnli', 'mrpc', 'rte', 'qnli', 'sst', 'wnli']
        task_dict = tasks.get_task_dict(task_names)
    elif args.tasks == 'arc':
        task_names = ['arc_easy', 'arc_challenge', 'hellaswag']
        task_dict = tasks.get_task_dict(task_names)
    elif args.tasks == 'all':
        task_names = ['mnli', 'mrpc', 'rte', 'qnli', 'sst', 'wnli', 'arc_easy', 'arc_challenge', 'hellaswag']
        task_dict = tasks.get_task_dict(task_names)
    elif args.tasks == 'lambada':
        task_names = ['lambada']
        task_dict = tasks.get_task_dict(task_names)
    else:
        raise NotImplementedError

    for task_name, task in task_dict.items():
        print(f"============= start task {task_name} =============")
        if not task.has_validation_docs():
            continue
        else:
            task.num_fewshot = args.num_fewshot
            task.provide_description = args.provide_description
            experiment.evaluation_task = task
            results = {**results, **experiment.evaluate(seeds=args.seeds)}

    dumped = json.dumps(results, indent=2)
    print(dumped)
    output_dir = os.path.join('down_stream_results', experiment.exp_name)
    print('save result in:', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{args.tasks}.json'), "w") as f:
        f.write(dumped)


if __name__ == "__main__":
    main()
