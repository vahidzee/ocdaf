import lightning.pytorch as pl
import typing as th
from jsonargparse import ActionConfigFile
from pprint import pprint
import os
from jsonargparse import Namespace, ArgumentParser
from jsonargparse.actions import ActionConfigFile
from pathlib import Path

from lightning.pytorch import LightningModule, LightningDataModule

from smart_trainer import change_config_for_causal_discovery
import traceback
from dysweep import dysweep_run_resume, ResumableSweepConfig

from lightning.pytorch.cli import LightningArgumentParser

from smart_trainer import convert_to_dict

def build_args():
    parser = ArgumentParser()
    parser.add_class_arguments(
        ResumableSweepConfig,
        fail_untyped=False,
        sub_configs=True,
    ) 
    parser.add_argument(
        "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
    )
    args = parser.parse_args()
    
    # set the lightning logger to true
    args.use_lightning_logger = True
    
    # set the default root dir to the current working directory
    args.default_root_dir = Path.cwd() / args.default_root_dir
    # if the path does not exists then create it
    if not os.path.exists(args.default_root_dir):
        os.makedirs(args.default_root_dir)

    return args


def run(conf, logger, checkpoint_dir):
    try:
        # TODO: remove this
        conf, _ = change_config_for_causal_discovery(
            conf, bypass_logger=True, 
        )
        
        # Add a lightning checkpointing callback for when we are trying to resume the last 
        # configuration. In this case, we will only keep the top 1 epoch.
        new_callback = {
            "class_path": "ocd.training.callbacks.checkpointing.DebuggedModelCheckpoint",
            "init_args": {
                "dirpath": checkpoint_dir,
                "verbose": True,
                "save_top_k": 1,
            },
        }
        
        # If a callback exists with that specification overwrite it, otherwise add it
        if len(conf["trainer"]["callbacks"]) > 0 and \
            conf['trainer']['callbacks'][-1]['class_path'] == "ocd.training.callbacks.checkpointing.DebuggedModelCheckpoint":
            conf["trainer"]["callbacks"][-1] = new_callback
        else:
            conf["trainer"]["callbacks"].append(new_callback)
        
        # Create a lightning parser to turn the configuration into a lightning configuration
        # we facilitate the instantiate_classses function that LightningCLI has provided
        # to turn all the init_args and class_path values into actual classes and arguments
        lightning_parser = LightningArgumentParser()
        lightning_parser.add_argument(
            "--seed_everything",
            type=th.Union[bool, int],
            default=True,
            help=(
                "Set to an int to run seed_everything with this value before classes instantiation."
                "Set to True to use a random seed."
            ),
        )
        lightning_parser.add_lightning_class_args(pl.Trainer, "trainer", required=False)
        lightning_parser.add_lightning_class_args(LightningModule, "model", subclass_mode=True, required=False)
        lightning_parser.add_lightning_class_args(LightningDataModule, "data", subclass_mode=True, required=False)
        config_init = lightning_parser.instantiate_classes(lightning_parser.parse_object(conf))
        
        # After getting the entire lightning init we then extract the model and datamodule, seed_everything
        # get the checkpoint if it exists and run the trainer
        model = config_init.model
        datamodule = config_init.data
        pl.seed_everything(config_init.seed_everything)

        # create the trainer, if logger existed in the config, then remove it and 
        # then create the trainer using the logger in the input
        if "logger" in config_init.trainer:
            config_init.trainer.pop("logger")
        trainer = pl.Trainer(logger=logger, **config_init.trainer)

        # Handle checkpointing
        ckpt_path = None
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            ckpt_path = checkpoint_dir / ckpt_list[-1]
        if ckpt_path is not None:
            print(">>>>> RUNNING WITH CHECKPOINT: ", ckpt_path)
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        
    except Exception as e:
        print("Exception:\n", e)
        print(traceback.format_exc())
        print("-----------")
        raise e


if __name__ == "__main__":
    args = build_args()
    dysweep_run_resume(args, run)