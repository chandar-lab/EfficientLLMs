import argparse
import json
import numpy as np
import random
import os

from lm_eval import tasks
from lm_eval.models.gpt2 import GPT2LM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', required=True)
    parser.add_argument('--tasks', default="all_tasks")
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output_path', default="SuperGLUE")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    lm = GPT2LM(args.configs, device='cuda')
    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names)
    results = {}
    for task_name, task in task_dict.items():
        print(f"============= start task {task_name} =============")
        if not task.has_validation_docs():
            continue
        result = task.evaluate(
            docs=task.validation_docs(),
            lm=lm,
            provide_description=args.provide_description,
            num_fewshot=args.num_fewshot,
        )
        results[task_name] = result
        print(result)

    dumped = json.dumps(results, indent=2)
    print(dumped)
    if args.output_path:
        output_dir = os.path.join('down_stream_results', lm.exp_name)
        print('save result in:', output_dir)
        os.makedirs(output_dir, exist_ok=True)
        #_{str(args.seed)}
        with open(os.path.join(output_dir, f'{args.output_path}.json'), "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    main()
