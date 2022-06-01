# -*- coding: UTF-8 -*-

import os
import subprocess
from sys import stdout
import pandas as pd
import argparse
import re
import traceback
import numpy as np
from typing import List


# Repeat experiments and save results to csv
# Example: python max_history.py --in_f run_max_history.sh --out_f max_history.csv --n 5


def parse_args():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument('--log_dir', nargs='?', default='../log/',
                        help='Log save dir.')
    parser.add_argument('--cmd_dir', nargs='?', default='./',
                        help='Command dir.')
    parser.add_argument('--in_f', nargs='?', default='run.sh',
                        help='Input commands.')
    parser.add_argument('--out_f', nargs='?', default='exp.csv',
                        help='Output csv.')
    parser.add_argument('--base_seed', type=int, default=0,
                        help='Random seed at the beginning.')
    parser.add_argument('--n', type=int, default=5,
                        help='Repeat times of each command.')
    parser.add_argument('--skip', type=int, default=0,
                        help='skip number.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    return parser.parse_args()


def find_info(result: List[str]) -> dict:
    info = dict()
    prefix = ''
    for line in result:        
        if line.startswith(prefix + 'Test After Training:'):
            p = re.compile('\(([\w@:\.\d,]+)\)')
            metrics = p.search(line).group(1)
            for m in metrics.split(','):
                metric_name, metric_value = m.split(':')
                info[metric_name] = float(metric_value)
    return info


def main():
    args = parse_args()
    columns = ['Model', 'Dataset', 'Max_history', 'HR@5', 'HR@10', 'HR@20', 'HR@50', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'NDCG@50']
    skip = args.skip
    df = pd.DataFrame(columns=columns)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    in_f = open(os.path.join(args.cmd_dir, args.in_f), 'r')
    lines = in_f.readlines()

    # Iterate commands
    for cmd in lines:
        cmd = cmd.strip()
        if cmd == '' or cmd.startswith('#') or cmd.startswith('export') or cmd.startswith('for') or cmd == 'do' or cmd == 'done':
            continue
        p = re.compile('--model_name (\w+)')
        model_name = p.search(cmd).group(1)        
        p = re.compile('--dataset (\w+)')
        dataset = p.search(cmd).group(1)
        p = re.compile('--history_max (\w+)')
        history_max = p.search(cmd).group(1)
                
        # Repeat experiments
        for i in range(args.base_seed, args.base_seed + args.n):
            try:
                command = cmd
                if command.find(' --random_seed') == -1:
                    command += ' --random_seed ' + str(i)
                if command.find(' --gpu ') == -1:
                    command += ' --gpu ' + args.gpu
                print(command)
                if skip > 0:
                    skip -= 1
                    continue                
                result = subprocess.run(command, shell=True, capture_output=True, text=True).stdout
                print(result)
                result = [line.strip() for line in result.split(os.linesep)]
                info = find_info(result)
                info['Model'] = model_name                              
                info['Dataset'] = dataset  
                info['Max_history'] = history_max              
                row = [info[c] if c in info else '' for c in columns]
                df.loc[len(df)] = row
                df.to_csv(os.path.join(args.log_dir, args.out_f), index=False)                
            except Exception as e:
                traceback.print_exc()
                continue

        # Average results
        if args.n > 1:
            info = {'Model': model_name}
            tests = df['Test'].tolist()[-args.n:]
            tests = [[float(m.split(':')[1]) for m in t.split(',')] for t in tests]
            avgs = ['{:<.4f}'.format(np.average([t[mi] for t in tests])) for mi in range(len(tests[0]))]
            info['Test'] = ','.join(avgs)
            row = [info[c] if c in info else '' for c in columns]
            df.loc[len(df)] = row
            
        # Save results
        df.to_csv(os.path.join(args.log_dir, args.out_f), index=False)


if __name__ == '__main__':
    main()
