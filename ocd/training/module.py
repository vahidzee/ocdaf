import typing as th
import torch
from lightning_toolbox import TrainingModule
import functools
import dycode as dy
import matplotlib.pyplot as plt
import numpy as np
import normflows as nf
import dycode as dy
from lightning.pytorch.utilities.types import EPOCH_OUTPUT
from ocd.evaluation import backward_score, count_backward


class OrderedTrainingModule(TrainingModule):
    def __init__(
        self,
        # (1) Model parameters:
        # essential flow args
        base_distribution: th.Union[nf.distributions.BaseDistribution, str],
        base_distribution_args: dict,
        # architecture
        in_features: th.Union[th.List[int], int],
        layers: th.List[th.Union[th.List[int], int]] = None,
        elementwise_perm: bool = False,
        residual: bool = False,
        bias: bool = True,
        activation: th.Optional[str] = "torch.nn.LeakyReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = False,
        batch_norm_args: th.Optional[dict] = None,
        # additional flow args
        additive: bool = False,
        num_transforms: int = 1,
        # ordering
        ordering: th.Optional[torch.IntTensor] = None,
        learn_permutation: bool = True,
        # general args
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        # (2) Trainer parameters:
        # Phase switch controls
        starting_phase: th.Literal["expectation", "maximization"] = "maximization",
        phase_change_upper_bound: int = 100,
        expectation_epoch_limit: int = 10,
        maximization_epoch_limit: int = 10,
        # switching context parameters
        overfit_window_size=10,
        overfit_check_threshold=0.001,
        overfitting_patience=10,
        # (3) criterion
        criterion_args: th.Optional[dict] = None,
        # optimization configs [is_active(training_module, optimizer_idx) -> bool]
        # (4) Optimizer parameters:
        optimizer: th.Union[str, th.List[str]] = "torch.optim.Adam",
        optimizer_is_active: th.Optional[th.Union[dy.FunctionDescriptor, th.List[dy.FunctionDescriptor]]] = None,
        optimizer_parameters: th.Optional[th.Union[th.List[str], str]] = None,
        optimizer_args: th.Optional[dict] = None,
        # learning rate
        lr: th.Union[th.List[float], float] = 1e-4,
        # schedulers
        scheduler: th.Optional[th.Union[str, th.List[str]]] = None,
        scheduler_name: th.Optional[th.Union[str, th.List[str]]] = None,
        scheduler_optimizer: th.Optional[th.Union[int, th.List[int]]] = None,
        scheduler_args: th.Optional[th.Union[dict, th.List[dict]]] = None,
        scheduler_interval: th.Union[str, th.List[str]] = "epoch",
        scheduler_frequency: th.Union[int, th.List[int]] = 1,
        scheduler_monitor: th.Optional[th.Union[str, th.List[str]]] = None,
    ):
        # initialize model and optimizer/scheduler configs
        _criterion_args = dict(terms=["ocd.training.terms.TrainingTerm"])
        _criterion_args.update(criterion_args or {})
        super().__init__(
            model_cls="ocd.models.ocd.OCD",
            model_args=dict(
                # essential flow args
                base_distribution=base_distribution,
                base_distribution_args=base_distribution_args,
                # architecture
                in_features=in_features,
                layers=layers,
                elementwise_perm=elementwise_perm,
                residual=residual,
                bias=bias,
                activation=activation,
                activation_args=activation_args,
                batch_norm=batch_norm,
                batch_norm_args=batch_norm_args,
                # additional flow args
                additive=additive,
                num_transforms=num_transforms,
                # ordering
                ordering=ordering,
                learn_permutation=learn_permutation,
                # general args
                device=device,
                dtype=dtype,
            ),
            # criterion
            criterion="lightning_toolbox.Criterion",
            criterion_args=_criterion_args,
            # training algorithm configs
            optimizer=optimizer,
            optimizer_is_active=optimizer_is_active,
            optimizer_parameters=optimizer_parameters,
            optimizer_args=optimizer_args,
            lr=lr,
            scheduler=scheduler,
            scheduler_name=scheduler_name,
            scheduler_optimizer=scheduler_optimizer,
            scheduler_args=scheduler_args,
            scheduler_interval=scheduler_interval,
            scheduler_frequency=scheduler_frequency,
            scheduler_monitor=scheduler_monitor,
            # hparams
            save_hparams=True,
        )

        # Phase switching:
        self.current_phase = starting_phase
        self.num_epochs_on_expectation = 0
        self.num_epochs_on_maximization = 0
        self.expectation_epoch_limit = expectation_epoch_limit
        self.maximization_epoch_limit = maximization_epoch_limit
        self.phase_change_rem = phase_change_upper_bound

        # overfitting checks
        self.overfit_window_size = overfit_window_size
        self.overfit_check_threshold = overfit_check_threshold
        self.overfitting_patience = overfitting_patience
        self.overfitting_patience_rem = overfitting_patience

        # Keep a list of monitored validation losses
        self.last_monitored_validation_losses = []

    def add_monitoring_value(self, val):
        if (
            len(self.last_monitored_validation_losses) > 1
            and self.last_monitored_validation_losses[-1] > self.last_monitored_validation_losses[-2]
        ):
            # pop the last element
            self.last_monitored_validation_losses.pop()
        self.last_monitored_validation_losses.append(val)
        if len(self.last_monitored_validation_losses) > self.overfit_window_size:
            self.last_monitored_validation_losses.pop(0)

    def end_maximization(self):
        # End it if the current last monitored validation loss is greater than the average of the last
        # overfit_check_patience losses
        t = len(self.last_monitored_validation_losses)
        if (
            t > 0
            and self.last_monitored_validation_losses[-1]
            >= sum(self.last_monitored_validation_losses) / t - self.overfit_check_threshold
        ):
            # If this condition is satisfied multiple times consequetively, then end the maximization phase
            self.overfitting_patience_rem -= 1
            if self.overfitting_patience_rem == 0:
                self.overfitting_patience_rem = self.overfitting_patience
            return True
        self.overfitting_patience_rem = self.overfitting_patience
        return False

    def end_expectation(self):
        self.num_epochs_on_expectation += 1
        if self.num_epochs_on_expectation >= self.expectation_epoch_limit:
            self.num_epochs_on_expectation = 0
            return True
        return False

    def reset_switch_phase(self):
        # Things that happen when the training phase switches
        self.num_epochs_on_expectation = 0
        self.num_epochs_on_maximization = 0
        self.last_monitored_validation_losses = []

    def get_phase(self) -> th.Literal["expectation", "maximization"]:
        return self.current_phase

    def re_evaluate_phase(self):
        """
        This function controls the phase switching
        when the model overfits in the maximization phase, this means that the autoregressive model
        begins to work worse and worse on validation data, so we switch to the expectation phase,
        this ensures that for every configuration of the latent permutation model, the best possible
        autoregressive model is learned
        """
        if self.current_phase == "expectation":
            if self.end_expectation():
                self.reset_switch_phase()
                self.current_phase = "maximization"
                self.phase_change_rem -= 1
        elif self.current_phase == "maximization":
            if self.end_maximization():
                # if it is starting to overfit, then reset all the overfitting settings
                # and switch phase
                self.reset_switch_phase()
                self.current_phase = "expectation"
                self.phase_change_rem -= 1

        # early stop the training if self.phase_change_rem is 0
        if self.phase_change_rem == 0:
            self.trainer.should_stop = True

    def step(
        self,
        batch: th.Optional[th.Any] = None,
        batch_idx: th.Optional[int] = None,
        optimizer_idx: th.Optional[int] = None,
        name: str = "train",
        transformed_batch: th.Optional[th.Any] = None,
        transform_batch: bool = True,
        return_results: bool = False,
        return_factors: bool = False,
        log_results: bool = True,
        **kwargs,  # additional arguments to pass to the criterion and attacker
    ):
        if name == "val":
            ret = super().step(
                batch=batch,
                batch_idx=batch_idx,
                optimizer_idx=optimizer_idx,
                name=name,
                transformed_batch=transformed_batch,
                transform_batch=transform_batch,
                return_results=True,
                return_factors=return_factors,
                log_results=log_results,
                original_batch=batch,
                **kwargs,
            )

            return ret[0] if return_factors else ret
        return super().step(
            batch=batch,
            batch_idx=batch_idx,
            optimizer_idx=optimizer_idx,
            name=name,
            transformed_batch=transformed_batch,
            transform_batch=transform_batch,
            return_results=return_results,
            return_factors=return_factors,
            log_results=log_results,
            original_batch=batch,
            **kwargs,
        )

    def validation_epoch_end(self, outputs: th.Union[EPOCH_OUTPUT, th.List[EPOCH_OUTPUT]]) -> None:
        # When validation epoch has ended, take all the outputs from validation steps
        # extract the losses and stack them up in the monitored value function
        all_losses = []
        for output in outputs if isinstance(outputs, list) else [outputs]:
            if isinstance(output, dict):
                if "loss" in output:
                    all_losses.append(output["loss"].item())
        mean_loss = sum(all_losses) / len(all_losses)
        self.add_monitoring_value(mean_loss)

        self.log("metrics/epoch/val_loss", mean_loss, on_step=False, on_epoch=True)
        return super().validation_epoch_end(outputs)

    def on_train_epoch_end(self) -> None:
        # When training epoch has ended, check if we need to switch phase
        self.re_evaluate_phase()
        return super().on_train_epoch_end()

    def on_train_epoch_start(self) -> None:
        # check the predicted graph structure
        # get the predicted graph structure
        with torch.no_grad():
            # compare to the true graph structure
            # get datamodule
            pass
            # self.log("metrics/tau", self.model.tau, on_step=False, on_epoch=True)
            # self.log("metrics/n_iter", self.model.n_iter, on_step=False, on_epoch=True)
        return super().on_train_epoch_start()
