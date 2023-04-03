"""
This callback contains some functionalities to evaluate how good the flow performs
"""

# from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl
from lightning_toolbox import TrainingModule
import torch
from ocd.visualization.qqplot import qqplot
import typing as th
import os
import yaml
from lightning.pytorch import Trainer
import pandas as pd


class CheckpointingCallback(Callback):
    def __init__(self, checkpoint_address: str, checkpoint_name: str, freq: int = -1):
        super().__init__()
        self.checkpoint_address = checkpoint_address
        self.checkpoint_name = checkpoint_name
        self.freq = freq
        if self.freq == 0:
            raise ValueError("freq cannot be 0")

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        trainer.datamodule.data.samples.to_csv(os.path.join(self.addr, "data.csv"))

        return super().on_fit_start(trainer, pl_module)

    @property
    def addr(self):
        addr = os.path.join(self.checkpoint_address, self.checkpoint_name)
        # check if the checkpoint address exists
        if not os.path.exists(addr):
            os.mkdir(addr)
        return addr

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: TrainingModule) -> None:
        if self.freq > 0:
            model_checkpoint_addr = os.path.join(self.addr, f"{trainer.current_epoch}-model.ckpt")
            torch.save(pl_module.model.state_dict(), model_checkpoint_addr)
            if (trainer.current_epoch + 1) % self.freq != 1:
                prev_model = os.path.join(self.addr, f"{trainer.current_epoch - 1}-model.ckpt")
                # remove the previous model
                if os.path.exists(prev_model):
                    os.remove(prev_model)
        else:
            model_checkpoint_addr = os.path.join(self.addr, f"model.ckpt")
            torch.save(pl_module.model.state_dict(), model_checkpoint_addr)

        # save pl_module.hparams to yaml file
        hparams_addr = os.path.join(self.addr, "model_args.yaml")
        with open(hparams_addr, "w") as f:
            yaml.dump(pl_module.model_args, f)

        return super().on_train_epoch_end(trainer, pl_module)
