from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningModule, LightningDataModule
from lightning_toolbox import TrainingModule
from lightning.pytorch.callbacks import ModelCheckpoint
import os


def main():
    cli = LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        auto_configure_optimizers=False,
    )


def dysweep_compatible_run(conf, logger, checkpoint_dir):
    """
    This is a dysweep standard run function that takes in the configuration as a dictionary
    'conf' and the lightning 'logger' and a checkpoint directory 'checkpoint_dir' to store
    the mid-step results for resuming.
    
    We use the lightning checkpointing capabilities to checkpoint our entire training pipeline.
    """
    try:
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
    # delete config.yaml file if it exists
    #
    # This is due to an exception in the LightningCLI
    # that occurs when the config.yaml file already exists
    if os.path.exists("config.yaml"):
        os.remove("config.yaml")

    main()
