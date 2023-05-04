import argparse
from functools import partial
from pathlib import Path
import os, sys

import pandas as pd
import wandb
import yaml

from cam import CAM
from score import Score
from permutohedron import Permutohedron

sys.path.append("..")

_DIR = os.path.dirname(os.path.abspath(__file__))
_REAL_WORLD_DIR = os.path.join(_DIR, '../experiments/data/real-world/')
_SYNTHETIC_DIR = os.path.join(_DIR, '../experiments/data/synthetic/')

_RESULTS_FILE = 'baseline_results.csv'


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', required=False, type=str)
    parser.add_argument('--project', required=False, type=str)
    parser.add_argument('--baseline', default='CAM', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_type', default='synthetic', type=str)
    parser.add_argument('--data_num', default=0, type=int)
    parser.add_argument('--default_root_dir', type=str)
    args = parser.parse_args()

    args.default_root_dir = Path.cwd() / 'results' if args.default_root_dir is None else Path(args.default_root_dir)
    return args


def get_data_config(data_type, data_num):
    if data_type == 'synthetic':
        paths = Path(_SYNTHETIC_DIR).glob('*.yaml')
    elif data_type == 'syntren':
        paths = Path(_REAL_WORLD_DIR).glob('data-syntren-*.yaml')
    else:
        paths = Path(_REAL_WORLD_DIR).glob('data-sachs.yaml')
    paths = sorted(list(paths))
    data_path = paths[data_num]
    data_config = yaml.load(open(data_path, 'r'), Loader=yaml.FullLoader)['init_args']
    data_name = str(data_path).split("/")[-1].split(".")[0]
    return data_config, data_name


def save_results(args, results_dict):
    csv_file = os.path.join(args.default_root_dir, _RESULTS_FILE)
    if not os.path.exists(csv_file):
        pd.DataFrame([results_dict]).to_csv(csv_file, index=False)
    else:
        df = pd.read_csv(csv_file)
        df.loc[len(df.index)] = results_dict
        df.to_csv(csv_file, index=False)


def run_baseline(args, wandb_mode=None):
    wandb_mode = 'online' if wandb_mode is None else wandb_mode
    wandb.init(project=args.project, mode=wandb_mode)
    config = vars(args)
    config.update(wandb.config)

    data_config, data_name = get_data_config(args.data_type, args.data_num)
    linear = data_name.split("_")[0] == 'linear'
    baseline_cls = {'CAM': CAM, 'Score': Score, 'Permutohedron': Permutohedron}[args.baseline]
    baseline_args = {'CAM': {'linear': linear},
                     'Score': {},
                     'Permutohedron': {'linear': linear, 'seed': args.seed}}[args.baseline]

    log = {'name': data_name, 'baseline': args.baseline, 'seed': args.seed, 'linear': linear}
    print(log)
    wandb.log(log)
    dataset_args = data_config['dataset_args'] if 'dataset_args' in data_config else None
    baseline = baseline_cls(dataset=data_config['dataset'], dataset_args=dataset_args, **baseline_args)
    result = baseline.evaluate()
    final_results = {**log, **result}
    wandb.log(final_results)
    save_results(args, final_results)


if __name__ == '__main__':
    the_args = build_args()
    func_to_call = partial(run_baseline, args=the_args)

    if the_args.sweep_id and the_args.project:
        wandb.agent(the_args.sweep_id, project=the_args.project, count=1, function=func_to_call)
    else:
        func_to_call(wandb_mode='disabled')
