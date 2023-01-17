import typing as th
import torch
from lightning_toolbox import TrainingModule
import dypy as dy


class OrderedTrainingModule(TrainingModule):
    def __init__(
        self,
        # (1) Model parameters:
        # essential flow args
        base_distribution: th.Union["normflows.distributions.BaseDistribution", str],
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
        log_input_outputs: bool = False,
        # general args
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        # (2) Trainer parameters:
        # Phase switch controls
        starting_phase: th.Literal["expectation", "maximization"] = "maximization",
        # Whether to use soft on maximization
        use_soft_on_maximization: bool = True,
        # Total number of phases
        phase_change_upper_bound: int = 100,
        # The number of epochs in each phase
        expectation_epoch_upper_bound: int = 10000,
        expectation_epoch_lower_bound: int = 10,
        maximization_epoch_upper_bound: int = 10000,
        maximization_epoch_lower_bound: int = 10,
        # switching context parameters
        overfit_window_size=10,
        overfit_check_threshold=0.05,
        overfitting_patience=3,
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
        self.use_soft_on_maximization = use_soft_on_maximization

        self.num_epochs_on_expectation = 0
        self.num_epochs_on_maximization = 0

        self.expectation_epoch_upper_bound = expectation_epoch_upper_bound
        self.expectation_epoch_lower_bound = expectation_epoch_lower_bound
        self.maximization_epoch_upper_bound = maximization_epoch_upper_bound
        self.maximization_epoch_lower_bound = maximization_epoch_lower_bound

        self.phase_change_rem = phase_change_upper_bound

        # overfitting checks
        self.overfit_window_size = overfit_window_size
        self.overfit_check_threshold = overfit_check_threshold
        self.overfitting_patience = overfitting_patience
        self.overfitting_patience_rem = overfitting_patience

        # Keep a list of monitored validation losses
        self.last_monitored_validation_losses = []

        # logging
        self.log_input_outputs = log_input_outputs
        self.logged_input_outputs = None

    def _check_logging_enabled(self):
        if not self.log_input_outputs:
            raise Exception("Logging is not enabled. Set log_permutations to True to enable logging.")

    def clear_logged_input_outputs(self):
        self._check_logging_enabled()
        self.logged_input_outputs = None

    def get_logged_input_outputs(self):
        self._check_logging_enabled()
        return self.logged_input_outputs

    def log_new_input_outputs(self, res=th.Dict[str, torch.Tensor]):
        self._check_logging_enabled()
        if self.logged_input_outputs is None:
            self.logged_input_outputs = res
        else:
            # append the new permutations to the existing ones
            for key in self.logged_input_outputs.keys():
                self.logged_input_outputs[key] = torch.cat([self.logged_input_outputs[key], res[key]], dim=0)

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
        # add the number of times this function is called
        self.num_epochs_on_maximization += 1

        # if it is called less than the lower bound return false
        if self.num_epochs_on_maximization < self.maximization_epoch_lower_bound:
            return False

        # if it is called more than the upper bound set the counter to zero and return true
        if self.num_epochs_on_maximization >= self.maximization_epoch_upper_bound:
            self.overfitting_patience_rem = self.overfitting_patience
            self.num_epochs_on_maximization = 0
            return True

        # End it if the current last monitored validation loss is greater than the average of the last
        # overfit_check_patience losses, of course, there is a patience factor at place as well
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
                self.num_epochs_on_maximization = 0
                return True

        self.overfitting_patience_rem = self.overfitting_patience
        return False

    def end_expectation(self):
        self.num_epochs_on_expectation += 1
        if self.num_epochs_on_expectation >= self.expectation_epoch_upper_bound:
            self.num_epochs_on_expectation = 0
            return True
        return False

    def reset_switch_phase(self):
        # Things that happen when the training phase switches
        self.num_epochs_on_expectation = 0
        self.num_epochs_on_maximization = 0
        self.last_monitored_validation_losses = []
        self.phase_change_rem -= 1
        if self.phase_change_rem == 0:
            self.trainer.should_stop = True

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
        elif self.current_phase == "maximization":
            if self.end_maximization():
                # if it is starting to overfit, then reset all the overfitting settings
                # and switch phase
                self.reset_switch_phase()
                self.current_phase = "expectation"

    def on_validation_epoch_end(self) -> None:
        # When validation epoch has ended, check if we need to switch phase
        results = super().on_validation_epoch_end()
        self.add_monitoring_value(self.trainer.fit_loop.epoch_loop._get_monitor_value("loss/val"))
        return results

    def on_train_epoch_end(self) -> None:
        # When training epoch has ended, check if we need to switch phase
        results = super().on_train_epoch_end()
        self.re_evaluate_phase()
        return results
