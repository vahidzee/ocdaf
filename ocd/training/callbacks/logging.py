import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import typing as th
from collections import defaultdict
from lightning_toolbox.objective_function.objective import Objective
import torch
import dypy as dy
from lightning_toolbox import TrainingModule


class LoggingCallback(Callback):
    """
    This is a callback used for logging the input and outputs of the training process
    the logs from this callback are used to generate plots for the other visualizing callbacks.

    All the values that are latched in the criterion are logged in the all_logged_values dict
    """

    def __init__(
        self,
        evaluate_every_n_epochs: int = 1,
        # TODO: Change this into a method with dynamize
        evaluate_every_n_epoch_logic: th.Optional[str] = None,
        epoch_buffer_size: int = 1,
        log_training: bool = True,
        log_validation: bool = False,
    ) -> None:
        super().__init__()
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.epoch_buffer_size = epoch_buffer_size
        self.log_training = log_training
        self.log_validation = log_validation

        self.epoch_counter = 0

        self.all_logged_values = defaultdict(list)

        self.validation_batches_in_epoch = 0
        self.training_batches_in_epoch = 0

        if evaluate_every_n_epoch_logic is not None:
            self.evaluate_every_n_epoch_logic = dy.eval_function(evaluate_every_n_epoch_logic)
        else:
            self.evaluate_every_n_epoch_logic = None

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: TrainingModule, outputs: STEP_OUTPUT, batch: th.Any, batch_idx: int
    ) -> None:
        if self.log_training:
            self.training_batches_in_epoch += 1
            for key, item in pl_module.objective.latch.items():
                if isinstance(item, torch.Tensor):
                    self.all_logged_values[key].append(item.detach().cpu())
                else:
                    self.all_logged_values[key].append(item)
            self.all_logged_values["loss"].append(pl_module.objective.results_latch["loss"].detach().cpu())
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: th.Optional[STEP_OUTPUT],
        batch: th.Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.log_validation:
            self.validation_batches_in_epoch += 1
            for key, item in pl_module.objective.latch.items():
                if isinstance(item, torch.Tensor):
                    self.all_logged_values[key].append(item.detach().cpu())
                else:
                    self.all_logged_values[key].append(item)

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def check_should_evaluate(self, trainer: pl.Trainer, pl_module: TrainingModule) -> bool:
        if self.evaluate_every_n_epoch_logic is None:
            return self.epoch_counter % self.evaluate_every_n_epochs == 0
        return self.evaluate_every_n_epoch_logic(trainer, pl_module)

    def evaluate(self, trainer: pl.Trainer, pl_module: TrainingModule) -> None:
        """
        The actual function that would be called whenever the logging functionality is needed
        """
        raise NotImplementedError(
            "The evaluate function of a logging callback should be implemented by the child class"
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: TrainingModule) -> None:
        res = super().on_train_epoch_end(trainer, pl_module)
        self.epoch_counter += 1

        if self.epoch_counter > self.epoch_buffer_size:
            # iterate over all the keys and pop the first element
            for key in self.all_logged_values.keys():
                for _ in range(self.training_batches_in_epoch + self.validation_batches_in_epoch):
                    self.all_logged_values[key].pop(0)

        self.training_batches_in_epoch = 0
        self.validation_batches_in_epoch = 0

        if self.check_should_evaluate(trainer, pl_module):
            self.evaluate(trainer, pl_module)

        return res
