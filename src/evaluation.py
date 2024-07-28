import argparse
import json
import numpy as np
import random
import os

from lm_eval import tasks
import copy
from runtime import Runtime
from common import Params, load_and_create_experiment_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', required=True, default='./configs/llama_baseline.jsonnet')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 1234, 1337, 2024, 1357],
                        help='list of seed integers for random number generation')
    parser.add_argument('--tasks', default="ppl")
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1024)
    parser.add_argument('--chk', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    _config, exp_name = load_and_create_experiment_name(args.configs)
    _config['_wandb_logs'] = False
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
    elif args.tasks == 'ppl':
        task_names = ['wikitext_2', 'wikitext_103', 'ptb', '1bw']
        task_dict = tasks.get_task_dict(task_names)
    else:
        raise NotImplementedError

    for task_name, task in task_dict.items():
        print(f"============= start task {task_name} =============")
        if not task.has_validation_docs():
            continue
        else:
            experiment.evaluation_task = task
            if args.tasks == 'ppl':
                experiment.evaluation_task.stride = args.stride
                result = {experiment.evaluation_task.TASK_NAME: experiment.evaluate(chk=args.chk)}
                results = {**results, **result}
            else:
                experiment.evaluation_task.num_fewshot = args.num_fewshot
                experiment.evaluation_task.provide_description = args.provide_description
                acc = []
                for seed in args.seeds:
                    experiment.set_seed(seed)
                    result = experiment.evaluate(chk=args.chk)
                    acc.append(result['minor']['acc'])
                _results = {
                    experiment.evaluation_task.TASK_NAME: {'mean': np.array(acc).mean(), 'std': np.array(acc).std()}}
                results = {**results, **_results}

    dumped = json.dumps(results, indent=2)
    print(dumped)
    output_dir = os.path.join('down_stream_results', experiment.exp_name)
    print('save result in:', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{args.tasks}.json'), "w") as f:
        f.write(dumped)


if __name__ == "__main__":
    main()