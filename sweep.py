import wandb
import lightning.pytorch as pl
from ruamel import yaml
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.cli import LightningArgumentParser
import typing as th
from jsonargparse import ActionConfigFile
from lightning.pytorch import LightningModule, LightningDataModule
import functools
from dataclasses import dataclass
from smart_trainer import convert_to_dict
from pprint import pprint
import os
from jsonargparse import Namespace
from smart_trainer import change_config_for_causal_discovery
from datetime import timedelta
import copy

SEPARATOR = "__CUSTOM_SEPERATOR__"
IDX_INDICATOR = "__IDX__"
SWEEP_INDICATION = "sweep"
UNIQUE_NAME_IDENT = "unique_name"
VALUES_DISPLAY_NAME = "values_display_name"


# Create a sweep dataclass to store the sweep configuration
@dataclass
class SweepMetric:
    goal: str
    name: str


@dataclass
class SweepConfigurations:
    method: str
    metric: SweepMetric
    parameters: dict


@dataclass
class Sweep:
    project: str
    agent_run_args: dict
    sweep_configuration: SweepConfigurations
    run_name: th.Optional[str] = None
    checkpoint_interval: int = 3600
    default_root_dir: th.Union[Path, str] = "experiments/sweep"
    sweep_id: th.Optional[th.Union[int, str]] = None
    wandb_id_file_path: th.Optional[th.Union[Path, str]] = None
    run_when_instantiate: bool = False
    use_smart_trainer: bool = False


compression_mapping = {}
value_compression_mapping = {}


def compress_parameter_config(parameter_config):
    current_tri = {}

    global compression_mapping
    global value_compression_mapping

    # if unique identifiers are provided in the sweep config, use them
    for key, val in parameter_config.items():
        if isinstance(val, dict):
            inner_dict = val.copy()
            if UNIQUE_NAME_IDENT in inner_dict:
                compression_mapping[key] = inner_dict[UNIQUE_NAME_IDENT]
                inner_dict.pop(UNIQUE_NAME_IDENT)
            if VALUES_DISPLAY_NAME in inner_dict and "values" in inner_dict:
                new_values = []
                for idx, value in enumerate(inner_dict["values"]):
                    if inner_dict[VALUES_DISPLAY_NAME][idx] in value_compression_mapping:
                        raise Exception(
                            f"Value {inner_dict[VALUES_DISPLAY_NAME][idx]} is already used in the sweep config"
                        )
                    value_compression_mapping[inner_dict[VALUES_DISPLAY_NAME][idx]] = value
                    new_values.append(inner_dict[VALUES_DISPLAY_NAME][idx])
                inner_dict["values"] = new_values
                inner_dict.pop(VALUES_DISPLAY_NAME)
            parameter_config[key] = inner_dict

    all_keys = list(parameter_config.keys())
    for key in all_keys:
        to_path = key.split(SEPARATOR)
        # reverse to_path
        from_path = to_path[::-1]
        current_node = current_tri
        current_path = None
        for p in from_path:
            current_path = f"{p}.{current_path}" if current_path is not None else p
            if p not in current_node:
                current_node[p] = {}
                if key not in compression_mapping:
                    compression_mapping[key] = current_path
                break

    ret = {}
    for key, val in parameter_config.items():
        ret[compression_mapping[key]] = val
    return ret


def decompress_parameter_config(parameter_config):
    global compression_mapping
    global value_compression_mapping

    ret = {}
    decompression_mapping = {v: k for k, v in compression_mapping.items()}

    for key, val in parameter_config.items():
        t = val
        if isinstance(t, str) and t in value_compression_mapping:
            t = value_compression_mapping[t]
        ret[decompression_mapping[key]] = t
    return ret


def build_parser(with_sweep: bool = True):
    # Setup the parser without any instantiation of the class
    parser = LightningArgumentParser()

    # Setup a lightning seed everything argument for reproducing results
    parser.add_argument(
        "--seed_everything",
        type=th.Union[bool, int],
        default=True,
        help=(
            "Set to an int to run seed_everything with this value before classes instantiation."
            "Set to True to use a random seed."
        ),
    )

    if with_sweep:
        parser.add_class_arguments(
            Sweep,
            nested_key="sweep",
            fail_untyped=False,
            sub_configs=True,
        )
    # parser.add_argument("--sweep", action=ActionConfigFile, help="Path to sweep configuration file")

    parser.add_lightning_class_args(pl.Trainer, "trainer")

    parser.add_lightning_class_args(LightningModule, "model", subclass_mode=True)
    parser.add_lightning_class_args(LightningDataModule, "data", subclass_mode=True)
    return parser


def build_args():
    parser = build_parser()
    args = parser.parse_args()

    args.sweep.default_root_dir = Path.cwd() / args.sweep.default_root_dir / "checkpoints"

    # if the path does not exists then create it
    if not os.path.exists(args.sweep.default_root_dir):
        os.makedirs(args.sweep.default_root_dir)

    args.sweep.wandb_id_file_path = args.sweep.default_root_dir / "_wandb_runid.txt"

    return args


def init_or_resume_wandb_run(
    wandb_id_file_path: Path, project_name: th.Optional[str] = None, run_name: th.Optional[str] = None
):
    """Detect the run id if it exists and resume
    from there, otherwise write the run id to file.

    Returns the config, if it's not None it will also update it first

    NOTE:
        Make sure that wandb_id_file_path.parent exists before calling this function
    """
    # if the run_id was previously saved, resume from there
    if wandb_id_file_path.exists():
        resume_id = wandb_id_file_path.read_text()
        logger = WandbLogger(project=project_name, name=run_name, id=resume_id)
    else:
        # if the run_id doesn't exist, then create a new run
        # and write the run id the file
        logger = WandbLogger(project=project_name, name=run_name)
        wandb_id_file_path.write_text(str(logger.experiment.id))

    return logger


def unflatten_sweep_config(flat_conf: dict):
    conf = {}
    for key, val in flat_conf.items():
        path_to_key = key.split(SEPARATOR)
        cur = conf
        for path in path_to_key[:-1]:
            if path not in cur:
                cur[path] = {}
            cur = cur[path]
        cur[path_to_key[-1]] = val

    def fix_list_components(tree_dict: dict) -> th.Union[th.List, dict]:
        all_elements_are_indices = True
        for key, val in tree_dict.items():
            if not key.startswith(IDX_INDICATOR):
                all_elements_are_indices = False
        if all_elements_are_indices:
            ret = [None for _ in range(len(tree_dict))]
            for key, val in tree_dict.items():
                idx = int(key[len(IDX_INDICATOR) :])
                ret[idx] = fix_list_components(val) if isinstance(val, dict) else val
            # Check if ret has any None values
            if any([val is None for val in ret]):
                raise ValueError("Not all indices are filled in the list")
            return ret
        else:
            ret = {}
            for key, val in tree_dict.items():
                ret[key] = fix_list_components(val) if isinstance(val, dict) else val
            return ret

    return fix_list_components(conf)


def flatten_sweep_config(tree_conf: dict):
    global compression_mapping
    global value_compression_mapping

    # Define a flattening method for the tree
    def flatten_tree(tree_dict: th.Union[dict, th.List]) -> dict:
        ret = {}
        has_something_to_iterate_over = False
        if isinstance(tree_dict, list):
            for idx, val in enumerate(tree_dict):
                if isinstance(val, dict) or isinstance(val, list):
                    if SWEEP_INDICATION in val:
                        # pass a version of val without the sweep indication
                        t = val.copy()
                        t.pop(SWEEP_INDICATION)
                        ret[IDX_INDICATOR + str(idx)] = t
                        has_something_to_iterate_over = True
                    else:
                        flattened, has_something = flatten_tree(val)
                        if has_something:
                            has_something_to_iterate_over = True
                            for subkey, subval in flattened.items():
                                ret[SEPARATOR.join([IDX_INDICATOR + str(idx), subkey])] = subval
        elif isinstance(tree_dict, dict):
            if SWEEP_INDICATION in tree_dict:
                for key, val in tree_dict.items():
                    if key != SWEEP_INDICATION:
                        ret[key] = val
            else:
                for key, val in tree_dict.items():
                    if isinstance(val, dict) or isinstance(val, list):
                        if SWEEP_INDICATION in val:
                            t = val.copy()
                            t.pop(SWEEP_INDICATION)
                            ret[key] = t
                            has_something_to_iterate_over = True
                        else:
                            flattened, has_something = flatten_tree(val)
                            if has_something:
                                has_something_to_iterate_over = True
                                for subkey, subval in flattened.items():
                                    ret[SEPARATOR.join([key, subkey])] = subval
        return ret, has_something_to_iterate_over

    flattened = flatten_tree(tree_conf)[0]
    conf_parameters = {}
    for key, val in flattened.items():
        conf_parameters[key] = val
    return conf_parameters


# overwrite args recursively
def overwrite_args(args, sweep_config):
    for key, val in sweep_config.items():
        if isinstance(val, dict):
            if isinstance(args, Namespace):
                new_args = overwrite_args(getattr(args, key), val)
                setattr(args, key, new_args)
            elif isinstance(args, dict):
                new_args = overwrite_args(args[key], val)
                args[key] = new_args
        else:
            if isinstance(args, Namespace):
                setattr(args, key, val)
            elif isinstance(args, dict):
                args[key] = val
    return args


def sweep_run(args):
    # ----------
    # wandb logger
    # ----------
    logger = init_or_resume_wandb_run(
        wandb_id_file_path=args.sweep.wandb_id_file_path, project_name=args.sweep.project, run_name=args.sweep.run_name
    )
    sweep_config = logger.experiment.config
    sweep_config = decompress_parameter_config(sweep_config)
    sweep_config = unflatten_sweep_config(sweep_config)

    # overwrite the sweep configurations
    args = overwrite_args(args, sweep_config)
    if args.sweep.use_smart_trainer:
        new_conf, _ = change_config_for_causal_discovery(args, bypass_logger=True)
    else:
        new_conf = convert_to_dict(sweep_conf_dict)
        
    # TODO: Add checkpoint callback to the trainer
    checkpoint_dir = Path(args.sweep.default_root_dir) / str(logger.experiment.id)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    new_conf["trainer"]["callbacks"].append(
        {
            "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
            "init_args": {
                "dirpath": checkpoint_dir,
                "verbose": True,
                "train_time_interval": timedelta(seconds=args.sweep.checkpoint_interval),
                "save_last": True,
                "save_top_k": -1,
            },
        }
    )

    args = overwrite_args(args, new_conf)

    parser = build_parser()
    args = parser.parse_object(args)

    # create a copy of args and drop the sweep configurations
    new_parser = build_parser(with_sweep=False)
    args_copy = copy.deepcopy(args)
    delattr(args_copy, "sweep")
    new_parser.save(
        args_copy,
        os.path.join(args.sweep.default_root_dir, f"discovery-config-{logger.experiment.id}.yaml"),
        skip_none=False,
        overwrite=True,
        multifile=False,
    )

    config_init = parser.instantiate_classes(args)

    model = config_init.model
    datamodule = config_init.data
    pl.seed_everything(config_init.seed_everything)

    # drop config_init.trainer.logger if it exists
    if "logger" in config_init.trainer:
        config_init.trainer.pop("logger")

    logger.experiment
    trainer = pl.Trainer(logger=logger, **config_init.trainer)

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    args = build_args()
    func_to_call = functools.partial(sweep_run, args=args)

    parameter_config = flatten_sweep_config(args.sweep.sweep_configuration.parameters)
    parameter_config = compress_parameter_config(parameter_config)
    # turn the args.sweep.sweep_configuration into a dictionary
    sweep_conf_dict = convert_to_dict(args.sweep.sweep_configuration)
    sweep_conf_dict["parameters"] = parameter_config

    if args.sweep.sweep_id is not None:
        sweep_id = args.sweep.sweep_id
        wandb.agent(sweep_id, function=func_to_call, project=args.sweep.project, **args.sweep.agent_run_args)
    else:
        sweep_id = wandb.sweep(sweep_conf_dict, project=args.sweep.project)
        if args.sweep.run_when_instantiate:
            wandb.agent(sweep_id, function=func_to_call, project=args.sweep.project, **args.sweep.agent_run_args)
