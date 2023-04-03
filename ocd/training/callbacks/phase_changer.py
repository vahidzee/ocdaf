from lightning.pytorch.callbacks import Callback
import typing as th
import lightning.pytorch as pl
from lightning_toolbox import TrainingModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch


class PhaseChangerCallback(Callback):
    """
    This callbacks sits on top of our main training module and controls the phase changing process

    pl_module.current_phase is the attribute that is being manipulated by this callback

    The phase change occures in different scenarios:

    (1) When the training process loss has converged to a certain value
    (2) When a generalization gap is occuring and the validation loss is increasing
    (3) When the number of epochs in each phase has reached their corresponding set limit

    For each of these settings, there are hyperparameters that can be adjusted to control
    the phase change process.

    """

    def __init__(
        self,
        starting_phase: th.Literal["maximization", "expectation"] = "maximization",
        # The setting for better performance
        check_every_n_iterations: int = 1,
        # The settings regarding epoch limit values
        maximization_epoch_limit: int = 10,
        expectation_epoch_limit: int = 10,
        # The settings regarding the loss convergence values
        loss_convergence_patience: int = 5,
        loss_convergence_threshold_eps: float = 0.0001,
        # The settings regarding the generalization gap
        generalization_patience: int = 5,
        generalization_threshold_eps: float = 0.0001,
    ):
        self.check_every_n_iterations = check_every_n_iterations
        self.training_iteration_counter = 0
        self.validation_iteration_counter = 0

        self.starting_phase = starting_phase

        self.epochs_on_maximization = 0
        self.maximization_epoch_limit = maximization_epoch_limit

        self.epochs_on_expectation = 0
        self.expectation_epoch_limit = expectation_epoch_limit

        self.baseline_training_loss_patience = loss_convergence_patience
        self.training_loss_patience_remaining = loss_convergence_patience
        self.loss_convergence_threshold_eps = loss_convergence_threshold_eps
        self.running_minimum_training_loss = float("inf")

        self.baseline_generalization_patience = generalization_patience
        self.generalization_patience_remaining = generalization_patience
        self.generalization_threshold_eps = generalization_threshold_eps
        self.running_minimum_validation_loss = float("inf")

    def change_phase(self, pl_module: TrainingModule) -> None:
        # Change the current_phase of the training_module
        if pl_module.current_phase == "maximization":
            pl_module.current_phase = "expectation"
        elif pl_module.current_phase == "expectation":
            pl_module.current_phase = "maximization"

        # change the number of epochs to zero
        pl_module.epochs_on_expectation = 0
        pl_module.epochs_on_maximization = 0

        # Change the loss convergence values
        self.training_loss_patience_remaining = self.baseline_training_loss_patience
        self.running_minimum_training_loss = float("inf")

        # Change the generalization gap values
        self.generalization_patience_remaining = self.baseline_generalization_patience
        self.running_minimum_validation_loss = float("inf")

    def on_fit_start(self, trainer: pl.Trainer, pl_module: TrainingModule) -> None:
        pl_module.current_phase = self.starting_phase
        return super().on_fit_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: TrainingModule) -> None:
        if pl_module.current_phase == "maximization":
            self.epochs_on_maximization += 1
            if self.epochs_on_maximization == self.maximization_epoch_limit:
                self.change_phase(pl_module)
        elif pl_module.current_phase == "expectation":
            self.epochs_on_expectation += 1
            if self.epochs_on_expectation == self.expectation_epoch_limit:
                self.change_phase(pl_module)

        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: TrainingModule,
        outputs: th.Optional[STEP_OUTPUT],
        batch: th.Any,
        batch_idx: int,
    ) -> None:
        ret = super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.training_iteration_counter += 1
        if self.training_iteration_counter % self.check_every_n_iterations == 0:
            return ret

        outputs = pl_module.objective.results_latch

        if "loss" not in outputs:
            raise Exception(f"The training step must return a loss value but got the following instead:\n{outputs}")

        # If the current loss is less than (min - eps) then reset the patience
        # otherwise, decrement the patience and if the patience reaches zero, change the phase
        current_loss = outputs["loss"]
        if current_loss < self.running_minimum_training_loss - self.loss_convergence_threshold_eps:
            self.training_loss_patience_remaining = self.baseline_training_loss_patience
        else:
            self.training_loss_patience_remaining -= 1
            if self.training_loss_patience_remaining == 0:
                print(">>>>>>> Changing phase <<<<<<<")
                print(">>> Due to loss convergence <<<<")
                self.change_phase(pl_module)

        # Take the minimum of current loss and the running minimum
        self.running_minimum_training_loss = min(self.running_minimum_training_loss, current_loss)

        return ret

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
        self.validation_iteration_counter += 1
        if self.validation_iteration_counter % self.check_every_n_iterations == 0:
            return ret

        outputs = pl_module.objective.results_latch

        if "loss" not in outputs:
            raise Exception(f"The validation step must return a loss value but got the following instead:\n{outputs}")

        # If the current loss is less than (min + eps) then reset the patience
        # otherwise, decrement the patience and if the patience reaches zero, change the phase
        current_loss = outputs["loss"]
        if current_loss <= self.running_minimum_validation_loss + self.generalization_threshold_eps:
            self.generalization_patience_remaining = self.baseline_generalization_patience
        else:
            self.generalization_patience_remaining -= 1
            if self.generalization_patience_remaining == 0:
                print(">>>>>>> Changing phase <<<<<<<")
                print(">>> Due to validation early stopping <<<<")
                self.change_phase(pl_module)

        # Take the minimum of current loss and the running minimum
        self.running_minimum_validation_loss = min(self.running_minimum_validation_loss, current_loss)
        return ret
