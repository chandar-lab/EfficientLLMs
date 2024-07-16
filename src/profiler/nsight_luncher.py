import pandas as pd
import numpy as np
import subprocess
import argparse
import os


def report_each_kernel(path='./plot/report_toy_example_gpukernsum.csv', verbos=False):
    df = pd.read_csv(path)
    res = {'linear_layers': 0, 'elementwise_kernel': 0, 'Flash_attention': 0, 'layer_norm': 0, 'cast': 0, 'total': 0}
    for i, _ in enumerate(df['Name']):
        name = df['Name'][i]
        res['total'] += df['Avg (ns)'][i] / 1_000_000
        st = ''
        if 'LoadWithCast' in name or 'reduce_kernel' in name:
            st += 'cast'
            res['cast'] += df['Avg (ns)'][i] / 1_000_000
        elif 'ampere' in name or 'cutlass_' in name:
            st += 'linear_layers'
            res['linear_layers'] += df['Avg (ns)'][i] / 1_000_000
        elif 'elementwise_kernel' in name:
            st += 'elementwise_kernel'
            res['elementwise_kernel'] += df['Avg (ns)'][i] / 1_000_000
        elif 'Flash' in name:
            st += 'Flash_attention'
            res['Flash_attention'] += df['Avg (ns)'][i] / 1_000_000
        elif 'layer_norm' in name or 'GammaBeta' in name:
            st += 'layer_norm'
            res['layer_norm'] += df['Avg (ns)'][i] / 1_000_000
        if verbos:
            print(
                f"================================{st}x{df['Instances'][i]}================================> {df['Total Time (ns)'][i] / 1_000_000}")
            print(name)
    return res


def args_parser():
    parser = argparse.ArgumentParser(
        description='profile time',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_path', default='../save', type=str, help='save path')
    parser.add_argument('--batch_size', action='store_true', help='iterate over batch size')
    parser.add_argument('--seq_len', action='store_true', help='iterate over seq_len')
    parser.add_argument('--model', action='store_true', help='iterate over models')
    args = parser.parse_args()
    return args


# Run the function
if __name__ == '__main__':
    args = args_parser()

    if args.batch_size:
        linear_percentage = []
        batch_size = [4, 8, 16, 32, 64]
        for b in batch_size:
            profile_cmd = f"nsys profile -w true -t cuda -s cpu --capture-range=cudaProfilerApi " \
                          f"--capture-range-end stop -x true -f true -o output_FB " \
                          f"python nsight_profile.py --batch_size {b} --seq_len 1024 " \
                          f"--model gpt2-small --bf16 "
            # Execute the profiling command
            print("Starting the profiling process...")
            subprocess.run(profile_cmd, shell=True)
            print("Profiling completed.")

            # Define the stats command
            stats_cmd = "nsys stats --output report_output_FB --report gpukernsum --force-overwrite true output_FB.nsys-rep"
            # Execute the stats command
            print("Generating statistics report...")
            subprocess.run(stats_cmd, shell=True)
            print("Statistics report generated.")
            try:
                res = report_each_kernel(path='./report_output_FB_gpukernsum.csv')
                linear_percentage.append(res['linear_layers']/res['total'])
            except Exception as e:
                print(f"An error occurred while processing the file: {e}")

            print("Deleting old files ...")
            subprocess.run("rm -rf ./output_FB*", shell=True)
            subprocess.run("rm -rf ./report_output_FB_gpukernsum.csv", shell=True)

            print(f"batch size:{b} ===> {linear_percentage[-1]}")

        result = {'batch_size': batch_size, 'linear_percentage': linear_percentage}
        print(result)
        df = pd.DataFrame(result)
        df.to_csv('./save_batch_size.csv')

    elif args.model:
        linear_percentage = []
        models = ['gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xlarge']
        for model in models:
            profile_cmd = f"nsys profile -w true -t cuda -s cpu --capture-range=cudaProfilerApi " \
                          f"--capture-range-end stop -x true -f true -o output_FB " \
                          f"python nsight_profile.py --batch_size 32 --seq_len 1024 " \
                          f"--model {model} --bf16 "
            # Execute the profiling command
            print("Starting the profiling process...")
            subprocess.run(profile_cmd, shell=True)
            print("Profiling completed.")

            # Define the stats command
            stats_cmd = "nsys stats --output report_output_FB --report gpukernsum --force-overwrite true output_FB.nsys-rep"
            # Execute the stats command
            print("Generating statistics report...")
            subprocess.run(stats_cmd, shell=True)
            print("Statistics report generated.")
            try:
                res = report_each_kernel(path='./report_output_FB_gpukernsum.csv')
                linear_percentage.append(res['linear_layers'] / res['total'])
            except Exception as e:
                print(f"An error occurred while processing the file: {e}")

            print("Deleting old files ...")
            subprocess.run("rm -rf ./output_FB*", shell=True)
            subprocess.run("rm -rf ./report_output_FB_gpukernsum.csv", shell=True)

            print(f"model:{model} ===> {linear_percentage[-1]}")

        result = {'models': models, 'linear_percentage': linear_percentage}
        print(result)
        df = pd.DataFrame(result)
        df.to_csv('./save_model.csv')

    elif args.seq_len:
        linear_percentage = []
        seq_lens = [256, 512, 1024, 2048, 4096]
        for seq_len in seq_lens:
            profile_cmd = f"nsys profile -w true -t cuda -s cpu --capture-range=cudaProfilerApi " \
                          f"--capture-range-end stop -x true -f true -o output_FB " \
                          f"python nsight_profile.py --batch_size 32 --seq_len {seq_len} " \
                          f"--model gpt2-small --bf16 "
            # Execute the profiling command
            print("Starting the profiling process...")
            subprocess.run(profile_cmd, shell=True)
            print("Profiling completed.")

            # Define the stats command
            stats_cmd = "nsys stats --output report_output_FB --report gpukernsum --force-overwrite true output_FB.nsys-rep"
            # Execute the stats command
            print("Generating statistics report...")
            subprocess.run(stats_cmd, shell=True)
            print("Statistics report generated.")
            try:
                res = report_each_kernel(path='./report_output_FB_gpukernsum.csv')
                linear_percentage.append(res['linear_layers'] / res['total'])
            except Exception as e:
                print(f"An error occurred while processing the file: {e}")

            print("Deleting old files ...")
            subprocess.run("rm -rf ./output_FB*", shell=True)
            subprocess.run("rm -rf ./report_output_FB_gpukernsum.csv", shell=True)

            print(f"sequence length:{seq_len} ===> {linear_percentage[-1]}")

        result = {'seq_lens': seq_lens, 'linear_percentage': linear_percentage}
        print(result)
        df = pd.DataFrame(result)
        df.to_csv('./save_seq_len.csv')

    else:
        seq_lens = [256, 512, 1024, 2048, 4096, 8192]
        models = ['gpt2-small', 'gpt2-medium', 'gpt2-large']
        result = {}
        for model in models:
            result[model] = {}
            for seq_len in seq_lens:

                profile_cmd = f"nsys profile -w true -t cuda -s cpu --capture-range=cudaProfilerApi " \
                              f"--capture-range-end stop -x true -f true -o output_FB " \
                              f"python nsight_profile.py --batch_size 32 --seq_len {seq_len} " \
                              f"--model {model} --bf16 "
                # Execute the profiling command
                print("Starting the profiling process...")
                subprocess.run(profile_cmd, shell=True)
                print("Profiling completed.")

                # Define the stats command
                stats_cmd = "nsys stats --output report_output_FB --report gpukernsum --force-overwrite true output_FB.nsys-rep"
                # Execute the stats command
                print("Generating statistics report...")
                subprocess.run(stats_cmd, shell=True)
                print("Statistics report generated.")
                try:
                    res = report_each_kernel(path='./report_output_FB_gpukernsum.csv')
                    result[model][seq_len] = res
                    print(f"############ {model}: sequence length:{seq_len} ===> {res}")
                    # linear_percentage.append(res['linear_layers'] / res['total'])
                except Exception as e:
                    print(f"An error occurred while processing the file: {e}")
                    exit()

                print("Deleting old files ...")
                subprocess.run("rm -rf ./output_FB*", shell=True)
                subprocess.run("rm -rf ./report_output_FB_gpukernsum.csv", shell=True)

        print(result)
        df = pd.DataFrame(result)
        df.to_csv('./save_all_linear.csv')
