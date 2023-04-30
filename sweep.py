import wandb
import lightning.pytorch as pl
import argparse
from ruamel import yaml
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.cli import LightningArgumentParser

SEPARATOR = "__CUSTOM_SEPERATOR__"


def build_args(arg_defaults=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--project', required=True, type=str)
    parser.add_argument('--run_name', required=True, type=str)
    
    parser.add_argument('--sweep_parameter_configurations_path', required=True, type=str)
    parser.add_argument('--metric_to_optimize', type=str, default='metrics/best-count_backward')
    parser.add_argument('--sweep-method', type=str, default='random')
    parser.add_argument('--sweep_metric_goal', type=str, default='minimize')
    
    parser.add_argument('--checkpoint_interval', default=3600, type=int)
    parser.add_argument('--default_root_dir', default='experiments/sweep', type=str)
    parser.add_argument('--resume_from_checkpoint', type=str)
    
    parser.add_argument('--agent_run_args.count', type=int, default=10)
    
    args = parser.parse_args()

    args.default_root_dir = Path.cwd() / args.default_root_dir / 'checkpoints'
    args.wandb_id_file_path = args.default_root_dir / '_wandb_runid.txt'
    
    return args

def read_sweep_config(args):
    # Setup the sweep config
    cfg = {}
    cfg['method'] = args.sweep_method
    cfg['metric'] = {
        'goal': args.sweep_metric_goal,
        'name': args.metric_to_optimize
    }
    
    # read a yaml file for the configuration of the parameters
    with open(args.sweep_parameter_configurations_path, 'r') as f:
        cfg['parameters'] = yaml.safe_load(f)
    
    # Define a flattening method for the tree
    def flatten_tree(tree_dict: dict) -> dict:
        for key, val in tree_dict.items():
            if isinstance(val, dict):
                for subkey, subval in flatten_tree(val):
                    yield SEPARATOR.join([key, subkey]), subval
            else:
                yield key, val  
    
    for key, val in flatten_tree(cfg['parameters']):
        cfg['parameters'][key] = {'value': val}
    
    return cfg
    
    
def overwrite_config_with_sweep_values(base_config, sweep_config):
    for key, value in sweep_config.items():
        path_to_key = key.split(SEPARATOR)
        cur = base_config
        for path in path_to_key[:-1]:
            cur = cur[path]
        cur[path_to_key[-1]] = value
    return base_config


def handle_checkpoint(args):
    checkpoint_dir = Path(args.default_root_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    elif args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob('*.ckpt'), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])

    args.callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir, 
            verbose=True,
            auto_insert_metric_name=False,
            train_time_interval=timedelta(seconds=args.checkpoint_interval)
        )
    )

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
                             id=resume_id)
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        logger = WandbLogger(project=project_name, name=run_name)
        wandb_id_file_path.write_text(str(logger.experiment.id))

    return logger

def sweep_run(args):
    """
    (1) Create or resume an already existing logger
    (2) Handle checkpointing
    """
    logger = init_or_resume_wandb_run(wandb_id_file_path=args.wandb_id_file_path,
                                      project_name=args.project, run_name=args.run_name)
    sweep_config = logger.experiment.config
    
    # (1) Create a base configuration from the arguments that are passed
    # (2) Put the logger id in the constructor of the logger if it exists and we are resuming
    # (3) Overwrite the base configuration with the sweep configuration
    
    cli = LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
    )
    # Check if wandb logger is used, if so raise an exception
    if cli.trainer.logger is not None and isinstance(cli.trainer.logger, pl.loggers.WandbLogger):
        raise ValueError("Wandb logger is not supported in sweep mode, try re-writing the configuration")
    
    parser = cli.parser
    base_config = parser.parse_args()
    
    with open(args.sweep_parameter_configurations_path, 'r') as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        
    base_config = overwrite_config_with_sweep_values(base_config, wandb.config)
    
    # Overwrite the base configuration with the causal discovery configuration
    config_for_discovery, logger_name = change_config_for_causal_discovery(base_config)
    log_dir = f"experiments/smart-trainer-logs/{logger_name}"

    custom_run(config_for_discovery, phase="causal_discovery", log_dir=log_dir)
    wandb.finish()
    

if __name__ == '__main__':
    args = build_args()
    
    args.log_every_n_steps = 1
    sweep_config = read_sweep_config(args)
    
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    
    wandb.agent(sweep_id, function=sweep_run, **args.agent_run_args)
    