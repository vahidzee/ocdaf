import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import typing as th

from collections import defaultdict


class LoggingCallback(Callback):
    """
    This is a callback used for logging the input and outputs of the training process
    the logs from this callback are used to generate plots for the other visualizing callbacks.

    All the values that are latched in the criterion are logged in the all_logged_values dict
    """

    def __init__(self,
                 log_training: bool = True,
                 log_validation: bool = False,
                 clear_every_n_epoch: int = 1,
                 display_every_n_epoch: int = 1) -> None:
        super().__init__()
        self.log_training = log_training
        self.log_validation = log_validation
        self.clear_every_n_epoch = clear_every_n_epoch
        self.clearing_counter = 0

        self.displaying_rem = 0

        self.all_logged_values = defaultdict(list)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                           outputs: STEP_OUTPUT,
                           batch: th.Any,
                           batch_idx: int) -> None:
        if self.log_training:
            # iterate over all the key and values of pl_module.criterion.latch
            # and add them to the all_logged_values dict
            for key, item in pl_module.criterion.latch.items():
                self.all_logged_values[key].append(item)

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                                outputs: th.Optional[STEP_OUTPUT], batch: th.Any, batch_idx: int,
                                dataloader_idx: int) -> None:

        if self.log_validation:
            # iterate over all the key and values of pl_module.criterion.latch
            # and add them to the all_logged_values dict
            for key, item in pl_module.criterion.latch.items():
                self.all_logged_values[key].append(item)

        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def check_should_clear(self):
        """
        Adds the counter and if it is greater than the clear_every_n_epoch
        it would return True
        """
        self.clearing_counter += 1
        if self.clearing_counter >= self.clear_every_n_epoch:
            self.clearing_counter = 0
            return True
        return False

    def check_should_display(self):
        return self.displaying_rem == 0

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                           outputs: STEP_OUTPUT,
                           batch: th.Any,
                           batch_idx: int) -> None:
        res = super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.displaying_rem = (self.displaying_rem +
                               1) % self.display_every_n_epoch
        return res

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.check_should_clear():
            self.all_logged_values = defaultdict(list)
        return super().on_train_epoch_start(trainer, pl_module)
