#!/usr/bin/env python3

import argparse
import copy
import os
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from shutil import which
from typing import Dict, Union, Any, List, Tuple, Optional

def generate_config_name(config_string):
    config_names = []
    config_paths = [path.strip() for path in config_string.split(',')]
    for path in config_paths:
        parts = os.path.splitext(os.path.basename(path))[0].split('/')
        config_names.append(parts[0].replace('-', ''))
    result = '_'.join(config_names)
    return result

def make_executable(script_path):
    mode = os.stat(str(script_path)).st_mode
    mode |= (mode & 0o444) >> 2
    os.chmod(str(script_path), mode)

def save_and_make_executable(job_path, script):
    with open(job_path, "w") as f:
        f.write(script)
    make_executable(job_path)

def replace_env_vars(target_str: str):
    for key, value in os.environ.items():
        target_str = target_str.replace(f"${key}", value)

    return target_str


def get_queued_jobs() -> List[Tuple[str, str, str]]:
    user = os.environ.get("USER")
    cmd = f"squeue -u {user} -o %A,%j,%T --noheader"
    output = subprocess.check_output(shlex.split(cmd)).decode("utf-8")
    jobs = []
    for line in output.splitlines():
        job_id, job_name, state = line.split(",")
        launcher_id = job_name.split("_compute.sh")[0]
        jobs.append((job_id, launcher_id, state))
    return jobs

class SlurmComputingCluster:
    def __init__(
            self,
            configs_dir: str = "../configs/example.jsonnet",
            scripts_dir: str = "~/scratch/efficient_llm_pre/jobs_scripts/",
            shared_storage_dir: str = "$SCRATCH",
            compute_storage_dir: str = "$SLURM_TMPDIR",
            hf_home: str = "$SCRATCH/hf_home",
            wandb_offline: bool = False,
            transformers_offline: bool = False,
            hf_datasets_offline: bool = False,
            account: str = 'rrg-bengioy-ad',
    ):
        self.configs_dir = configs_dir
        self.exp_name = generate_config_name(configs_dir)
        self.global_logs_dir = './save'
        self.log_dir = Path(self.global_logs_dir) / self.exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.global_scripts_dir = Path(scripts_dir)
        self.script_dir = Path(self.global_scripts_dir) / f"{self.exp_name}_job.sh"
        self.script_dir.mkdir(parents=True, exist_ok=True)

        self.cluster_shared_storage_dir = Path(
            replace_env_vars(shared_storage_dir)
        ).expanduser()
        self.compute_node_storage_dir = compute_storage_dir

        self.wandb_offline = wandb_offline
        self.transformers_offline = transformers_offline
        self.hf_datasets_offline = hf_datasets_offline
        self.account = account
        self.hf_home = hf_home

    def prepare_job(self) -> str:
        pass

    def execute_job(self):
        pass


class ComputeCanadaCluster(SlurmComputingCluster):
    def __init__(
            self,
            time: int = 3,
            gres: int = 4,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.gres = gres
        self.time = time

    def creat_script(self):
        script = "#!/bin/bash\n"
        script += f"#SBATCH -o {self.log_dir}/compute_log.txt\n"
        script += f"#SBATCH --account={self.account}\n"
        # script = "#SBATCH --nodes=1\n"
        script += f"#SBATCH --gres=gpu:{self.gres}\n"
        script += "#SBATCH --cpus-per-task=8\n"
        script += "#SBATCH --mem=128G\n"
        script += f"#SBATCH --job-name={self.exp_name}\n"
        script += "#SBATCH --time=0-03:00\n\n"

        script += "# load modules\n"
        script += "module load python/3.10 gcc/9.3.0 git-lfs/3.3.0 rust/1.70.0 protobuf/3.21.3 cuda/11.8.0 cudnn/8.6.0.163 arrow/12.0.1 httpproxy\n\n"

        script += "# activate environment\n"
        script += "source ../ENV/bin/activate\n\n"

        script += f'echo "Copying HF dataset to slurm dir..."\n'
        new_hf_home = Path(self.compute_node_storage_dir) / "hf_home"
        new_hf_home.mkdir(parents=True, exist_ok=True)
        script += f"rsync -avz {self.hf_home} {new_hf_home}\n"
        script += f"export HF_HOME={new_hf_home}\n\n"

        script += "# configure\n"
        script += "export HF_HUB_OFFLINE=1\n"
        script += "export HF_DATASETS_OFFLINE=1\n"
        script += "export TRANSFORMERS_OFFLINE=1\n"
        script += "export HF_HUB_DISABLE_TELEMETRY=1\n"
        script += "export HF_HUB_ENABLE_HF_TRANSFER=1\n"
        script += "export CUBLAS_WORKSPACE_CONFIG=:16:8\n"
        if self.gres == 1:
            script += f"python src/main.py --configs {self.configs_dir} train"
        else:
            script += f"torchrun --nproc_per_node=4 src/main.py --configs {self.configs_dir} train"

        return script

    def execute_job(self):
        script = self.creat_script()
        save_and_make_executable(self.script_dir, script)

        # print("Started executing...")
        # print("To check all logs, visit this directory:")
        # print(f"$ cd {self.log_dir} && ls -lh")


def main(args):
    if args.platform == "cc":
        clstr = ComputeCanadaCluster(configs_dir=args.main_configs+args.qconfigs,
                                     account=args.account,
                                     gres=args.gres,
                                     time=args.time)
        if args.info:
            queued_jobs = get_queued_jobs()
            print(f"Queued jobs:")
            from pprint import pprint
            pprint(queued_jobs)

        clstr.execute_job()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment runner")

    parser.add_argument(
        "-p",
        "--platform",
        metavar="PLATFORM",
        type=str,
        choices=["mila", "cc", "local"],
        default="cc",
        help="The computation platform we're running the experiment",
    )

    parser.add_argument(
        "--gres",
        type=int,
        default=4,
        help="number of gpus",
    )

    parser.add_argument(
        "--time",
        type=int,
        default=3,
        help="scheduling time",
    )

    parser.add_argument(
        "--account",
        metavar="ACCOUNT",
        type=str,
        help="Slurm account (only needed for CC)",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Print queued experiments' info",
        default=False,
    )

    parser.add_argument(
        "--qconfigs",
        "-q",
        type=str,
        default="configs/quantization/w4_sym_per_tensor.jsonnet",
        help="quantization config",
    )

    parser.add_argument(
        "--main_configs",
        type=str,
        default="configs/base.jsonnet, configs/model/gp2-small.jsonnet, configs/dataset/openwebtext.jsonnet, configs/trainer/gpt2_4gpu_ddp_CC.jsonnet",
        help="main configs",
    )

    args = parser.parse_args()

    main(args)
