import os, sys, argparse
from typing import Optional, Dict

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../model'))
sys.path.append(os.path.join(dir_path, '../data'))
from utils import get_diam_bounds

import numpy as np
from collections import OrderedDict
from load_dag import gen_params
from load_data import *
from sinkhorn_gn import SinkhornGN
from common import LagrangeStart, MetricsCallback, LitProgressBar
from estimands import *

from ACIC.datamodule import ACICDataModule

import wandb
from pytorch_lightning import Trainer
import pickle
from sklearn.model_selection import ParameterGrid
from load_dag import identifiable, non_identifiable

from pathlib import Path
from evaluations import acic_total_eval

from pytorch_lightning.loggers import WandbLogger

from functools import partial

import pytorch_lightning as pl

from datetime import timedelta

AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = 4
import os, sys, argparse
from typing import Optional, Dict

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../model'))
sys.path.append(os.path.join(dir_path, '../data'))
from utils import get_diam_bounds

import numpy as np
from collections import OrderedDict
from load_dag import gen_params
from load_data import *
from sinkhorn_gn import SinkhornGN
from common import LagrangeStart, MetricsCallback, LitProgressBar
from estimands import *

from ACIC.datamodule import ACICDataModule

import wandb
from pytorch_lightning import Trainer
import pickle
from sklearn.model_selection import ParameterGrid
from load_dag import identifiable, non_identifiable

from pathlib import Path
from evaluations import acic_total_eval

from pytorch_lightning.loggers import WandbLogger

from functools import partial

import pytorch_lightning as pl

from datetime import timedelta

AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = 4

SEED = 73532


def handle_checkpoint(args):
    checkpoint_dir = Path(args.default_root_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    elif args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob('*.ckpt'), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    args.callbacks.append(
        pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, verbose=True,
                                     auto_insert_metric_name=False,
                                     train_time_interval=timedelta(seconds=args.checkpoint_interval))
    )

    return args


def build_args(arg_defaults=None):
    pl.seed_everything(SEED)
    tmp = arg_defaults
    arg_defaults = {
        'max_epochs': 300,
        'gpus': AVAIL_GPUS,
        'num_workers': NUM_WORKERS,
        'batch_size': 4096,
        'callbacks': [],
    }
    if tmp is not None:
        arg_defaults.update(tmp)

    parser = argparse.ArgumentParser()
    # parser.add_argument('-do_var', '--dag_name', required=False, type=str,
    #                     choices=['backdoor', 'frontdoor', 'mdag', 'napkin', 'bow', 'extended_bow', 'iv', 'badm'])
    # parser.add_argument('--data_id', required=True, type=int)
    parser.add_argument('--sweep_id', required=True, type=str)
    parser.add_argument('--project', required=True, type=str)
    parser.add_argument('--run_name', required=True, type=str)
    parser.add_argument('--checkpoint_interval', default=3600, type=int)
    parser.add_argument('--default_root_dir', type=str)
    parser.add_argument('--resume_from_checkpoint', type=str)
    # parser.add_argument('--lagrange_lr', required=False, type=float)
    # parser = pl.Trainer.add_argparse_args(parser)
    parser = SinkhornGN.add_model_specific_args(parser)
    parser.set_defaults(**arg_defaults)
    args = parser.parse_args()

    if args.default_root_dir is None:
        args.default_root_dir = Path.cwd() / 'checkpoints'
    else:
        args.default_root_dir = Path(args.default_root_dir)

    return args


def init_or_resume_wandb_run(wandb_id_file_path: Path,
                             project_name: Optional[str] = None,
                             run_name: Optional[str] = None):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file.

        Returns the config, if it's not None it will also update it first

        NOTE:
            Make sure that wandb_id_file_path.parent exists before calling this function
    """
    # if the run_id was previously saved, resume from there
    if wandb_id_file_path.exists():
        resume_id = wandb_id_file_path.read_text()
        logger = WandbLogger(project=project_name,
                             name=run_name,
                             resume=resume_id)
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        logger = WandbLogger(project=project_name, name=run_name)
        wandb_id_file_path.write_text(str(logger.experiment.id))

    return logger


def run_expt(args):
    # ----------
    # wandb logger
    # ----------
    logger = init_or_resume_wandb_run(wandb_id_file_path=args.wandb_id_file_path,
                                      project_name=args.project, run_name=args.run_name)

    config = vars(args)
    config.update(logger.experiment.config)
    args.logger = logger

    # the same hidden/layers for generator and disc (TODO redo it)
    args.n_hidden = args.n_hidden
    args.n_layers = args.n_layers

    name, graph, do_var, target_var, n_latent, latent_dim = gen_params('backdoor')
    data_args = {'linear': True, 'n_samples': 5000, 'batch_size': args.batch_size, 'num_workers': NUM_WORKERS,
                 'validation_size': 0.1}

    res = gen_data(name, data_args)
    data, dm, var_dims, true_atd = res['data'], res['dm'], res['var_dims'], res['true_atd']
    diam, lower_bound, upper_bound = get_diam_bounds(data, var_dims)

    param_fn = create_atd(do_var, target_var, var_dims, delta=0.1)

    model = SinkhornGN(param_fn, graph, var_dims, n_latent, latent_dim,
                       upper_bounds=upper_bound, lower_bounds=lower_bound,
                       diameter=diam,
                       n_hidden=args.n_hidden, n_layers=args.n_layers,
                       lr=args.lr,
                       lagrange_lr=args.lagrange_lr, )

    metrics_callback = MetricsCallback()
    lagrange_start = LagrangeStart(monitor='val_dist', min_delta=0.001,
                                   patience=30, verbose=True, mode='min')
    prog_bar = LitProgressBar(refresh_rate=20)
    args.callbacks.extend([metrics_callback, prog_bar, lagrange_start])
    trainer = Trainer.from_argparse_args(args, log_every_n_steps=5)

    trainer.fit(model, dm)

    rdir = Path(args.default_root_dir)
    fname = os.path.join(rdir, args.run_name + '.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(metrics_callback.metrics, f)

    # acic_total_eval(model_joint, dm, args.data_id, target_vars, do_var, var_dims,
    #                 f'/h/vdblm/projects/NCM/neural_sems/outputs/{args.data_id}.csv')


if __name__ == '__main__':
    # ----------
    # args
    # ----------
    args = build_args()

    # ----------
    # checkpoints
    # ----------
    args = handle_checkpoint(args)

    # ----------
    # sweep agent
    # ----------
    args.wandb_id_file_path = args.default_root_dir / '_wandb_runid.txt'
    args.log_every_n_steps = 1

    func_to_call = partial(run_expt, args=args)

    func_to_call()

    if args.wandb_id_file_path.exists():
        func_to_call()
    else:
        wandb.agent(args.sweep_id, project=args.project, count=1, function=func_to_call)

    # python train.py --n_hidden_g 10 --n_layers_g 1 --n_hidden_d 10 --n_layers_d 0 --latent_dim 1

SEED = 73532


def handle_checkpoint(args):
    checkpoint_dir = Path(args.default_root_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    elif args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob('*.ckpt'), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    args.callbacks.append(
        pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir, verbose=True,
                                     auto_insert_metric_name=False,
                                     train_time_interval=timedelta(seconds=args.checkpoint_interval))
    )

    return args


def build_args(arg_defaults=None):
    pl.seed_everything(SEED)
    tmp = arg_defaults
    arg_defaults = {
        'max_epochs': 300,
        'gpus': AVAIL_GPUS,
        'num_workers': NUM_WORKERS,
        'batch_size': 4096,
        'callbacks': [],
    }
    if tmp is not None:
        arg_defaults.update(tmp)

    parser = argparse.ArgumentParser()
    # parser.add_argument('-do_var', '--dag_name', required=False, type=str,
    #                     choices=['backdoor', 'frontdoor', 'mdag', 'napkin', 'bow', 'extended_bow', 'iv', 'badm'])
    # parser.add_argument('--data_id', required=True, type=int)
    parser.add_argument('--sweep_id', required=True, type=str)
    parser.add_argument('--project', required=True, type=str)
    parser.add_argument('--run_name', required=True, type=str)
    parser.add_argument('--checkpoint_interval', default=3600, type=int)
    parser.add_argument('--default_root_dir', type=str)
    parser.add_argument('--resume_from_checkpoint', type=str)
    # parser.add_argument('--lagrange_lr', required=False, type=float)
    # parser = pl.Trainer.add_argparse_args(parser)
    parser = SinkhornGN.add_model_specific_args(parser)
    parser.set_defaults(**arg_defaults)
    args = parser.parse_args()

    if args.default_root_dir is None:
        args.default_root_dir = Path.cwd() / 'checkpoints'
    else:
        args.default_root_dir = Path(args.default_root_dir)

    return args


def init_or_resume_wandb_run(wandb_id_file_path: Path,
                             project_name: Optional[str] = None,
                             run_name: Optional[str] = None):
    """Detect the run id if it exists and resume
        from there, otherwise write the run id to file.

        Returns the config, if it's not None it will also update it first

        NOTE:
            Make sure that wandb_id_file_path.parent exists before calling this function
    """
    # if the run_id was previously saved, resume from there
    if wandb_id_file_path.exists():
        resume_id = wandb_id_file_path.read_text()
        logger = WandbLogger(project=project_name,
                             name=run_name,
                             resume=resume_id)
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        logger = WandbLogger(project=project_name, name=run_name)
        wandb_id_file_path.write_text(str(logger.experiment.id))

    return logger


def run_expt(args):
    # ----------
    # wandb logger
    # ----------
    logger = init_or_resume_wandb_run(wandb_id_file_path=args.wandb_id_file_path,
                                      project_name=args.project, run_name=args.run_name)

    config = vars(args)
    config.update(logger.experiment.config)
    args.logger = logger

    # the same hidden/layers for generator and disc (TODO redo it)
    args.n_hidden = args.n_hidden
    args.n_layers = args.n_layers

    name, graph, do_var, target_var, n_latent, latent_dim = gen_params('backdoor')
    data_args = {'linear': True, 'n_samples': 5000, 'batch_size': args.batch_size, 'num_workers': NUM_WORKERS,
                 'validation_size': 0.1}

    res = gen_data(name, data_args)
    data, dm, var_dims, true_atd = res['data'], res['dm'], res['var_dims'], res['true_atd']
    diam, lower_bound, upper_bound = get_diam_bounds(data, var_dims)

    param_fn = create_atd(do_var, target_var, var_dims, delta=0.1)

    model = SinkhornGN(param_fn, graph, var_dims, n_latent, latent_dim,
                       upper_bounds=upper_bound, lower_bounds=lower_bound,
                       diameter=diam,
                       n_hidden=args.n_hidden, n_layers=args.n_layers,
                       lr=args.lr,
                       lagrange_lr=args.lagrange_lr, )

    metrics_callback = MetricsCallback()
    lagrange_start = LagrangeStart(monitor='val_dist', min_delta=0.001,
                                   patience=30, verbose=True, mode='min')
    prog_bar = LitProgressBar(refresh_rate=20)
    args.callbacks.extend([metrics_callback, prog_bar, lagrange_start])
    trainer = Trainer.from_argparse_args(args, log_every_n_steps=5)

    trainer.fit(model, dm)

    rdir = Path(args.default_root_dir)
    fname = os.path.join(rdir, args.run_name + '.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(metrics_callback.metrics, f)

    # acic_total_eval(model_joint, dm, args.data_id, target_vars, do_var, var_dims,
    #                 f'/h/vdblm/projects/NCM/neural_sems/outputs/{args.data_id}.csv')


if __name__ == '__main__':
    # ----------
    # args
    # ----------
    args = build_args()

    # ----------
    # checkpoints
    # ----------
    args = handle_checkpoint(args)

    # ----------
    # sweep agent
    # ----------
    args.wandb_id_file_path = args.default_root_dir / '_wandb_runid.txt'
    args.log_every_n_steps = 1

    func_to_call = partial(run_expt, args=args)

    # func_to_call()

    if args.wandb_id_file_path.exists():
        func_to_call()
    else:
        wandb.agent(args.sweep_id, project=args.project, count=1, function=func_to_call)

    # python train.py --n_hidden_g 10 --n_layers_g 1 --n_hidden_d 10 --n_layers_d 0 --latent_dim 1