from lightning.pytorch.callbacks import Callback
import typing as th
import lightning.pytorch as pl
from lightning_toolbox import TrainingModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from lightning.pytorch import Trainer
from lightning_toolbox import TrainingModule


class PhaseChangerCallback(Callback):
    """
    This callbacks sits on top of our main training module and controls the phase changing process

    pl_module.current_phase is the attribute that is being manipulated by this callback

    The phase change occures in different scenarios:

    (1) When the training process loss has converged to a certain value
    (2) When the monitoring value (in this case loss/validation loss) has Plateaued

    For each of these settings, there are hyperparameters that can be adjusted to control
    the phase change process.

    """

    def __init__(
        self,
        starting_phase: th.Literal["maximization", "expectation"] = "maximization",
        # The setting for better performance
        # The settings regarding epoch limit values
        maximization_epoch_limit: int = 10,
        expectation_epoch_limit: int = 10,
        # The settings regarding the loss convergence values
        check_every_n_iterations: int = 1,  # for performance reasons
        patience: int = 5,
        threshold: float = 0.0001,
        cooldown: int = 0,
        reset_optimizers: bool = True,
        reinitialize_weights_on_maximization: bool = False,
        log_onto_logger: bool = True,
    ):
        self.check_every_n_iterations = check_every_n_iterations
        self.training_iteration_counter = 0
        self.validation_iteration_counter = 0

        self.starting_phase = starting_phase

        self.epochs_on_maximization = 0
        self.maximization_epoch_limit = maximization_epoch_limit

        self.epochs_on_expectation = 0
        self.expectation_epoch_limit = expectation_epoch_limit

        self.baseline_patience = patience
        self.patience_remaining = patience
        self.threshold = threshold
        self.validation_running_avg = 0
        self.running_minimum_validation_loss = float("inf")
        self.cooldown = cooldown
        self.num_validation_batches = 0
        self.cooldown_counter = cooldown
        self.cooldown_base_counter = cooldown

        self.reset_optimizers = reset_optimizers
        self.reinitialize_weights_on_maximization = reinitialize_weights_on_maximization

        self.log_onto_logger = log_onto_logger

    def change_phase(self, trainer: pl.Trainer, pl_module: TrainingModule) -> None:
        # Change the current_phase of the training_module
        if pl_module.current_phase == "maximization":
            pl_module.current_phase = "expectation"
        elif pl_module.current_phase == "expectation":
            pl_module.current_phase = "maximization"

        # change the number of epochs to zero
        self.epochs_on_expectation = 0
        self.epochs_on_maximization = 0

        # Change the generalization gap values
        self.running_minimum_validation_loss = float("inf")
        self.patience_remaining = self.baseline_patience

        if self.reset_optimizers:
            pl_module.reset_optimizers()

        if self.reinitialize_weights_on_maximization and pl_module.current_phase == "maximization":
            pl_module.reinitialize_flow_weights()

        self.cooldown_counter = self.cooldown_base_counter

    def on_fit_start(self, trainer: pl.Trainer, pl_module: TrainingModule) -> None:
        pl_module.current_phase = self.starting_phase
        return super().on_fit_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: TrainingModule) -> None:
        if pl_module.current_phase == "maximization":
            self.epochs_on_maximization += 1
            if self.epochs_on_maximization == self.maximization_epoch_limit:
                self.change_phase(trainer, pl_module)
        elif pl_module.current_phase == "expectation":
            self.epochs_on_expectation += 1
            if self.epochs_on_expectation == self.expectation_epoch_limit:
                self.change_phase(trainer, pl_module)

        return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: TrainingModule,
        outputs: th.Optional[STEP_OUTPUT],
        batch: th.Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        ret = super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

        outputs = pl_module.objective.results_latch

        if "loss" not in outputs:
            raise Exception(f"The validation step must return a loss value but got the following instead:\n{outputs}")

        # If the current loss is less than (min + eps) then reset the patience
        # otherwise, decrement the patience and if the patience reaches zero, change the phase
        self.validation_running_avg = (self.validation_running_avg * batch_idx + outputs["loss"]) / (batch_idx + 1)
        if self.num_validation_batches < batch_idx + 1:
            self.num_validation_batches = batch_idx + 1
        else:
            if self.num_validation_batches == batch_idx + 1:
                should_change_phase = False

                current_loss = self.validation_running_avg

                self.cooldown_counter = max(0, self.cooldown_counter - 1)

                if current_loss <= self.running_minimum_validation_loss * (1 - self.threshold):
                    self.patience_remaining = self.baseline_patience
                else:
                    self.patience_remaining = max(0, self.patience_remaining - 1)
                    if self.patience_remaining == 0 and self.cooldown_counter == 0:
                        should_change_phase = True

                # Take the minimum of current loss and the running minimum
                self.running_minimum_validation_loss = min(self.running_minimum_validation_loss, current_loss)

                if self.log_onto_logger:
                    pl_module.log("phase-changer/current_validation_loss", current_loss)
                    pl_module.log(
                        "phase-changer/running_minimum_validation_loss", self.running_minimum_validation_loss
                    )
                    pl_module.log("phase-changer/patience_remaining", float(self.patience_remaining))
                    pl_module.log(
                        "phase-changer/current_phase-0-maximization-1-expectation",
                        0.0 if pl_module.current_phase == "maximization" else 1.0,
                    )
                    pl_module.log("phase-changer/cooldown-counter", float(self.cooldown_counter))

                    if should_change_phase:
                        self.change_phase(trainer, pl_module)

        return ret
