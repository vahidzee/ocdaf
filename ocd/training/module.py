import typing as th
import torch
from lightning_toolbox import TrainingModule
import functools
import dycode as dy
from ocd.models.sinkhorn import sample_permutation_matrix
from ocd.evaluation import count_backward


class OrderedTrainingModule(TrainingModule):
    def __init__(
        self,
        # model
        in_covariate_features: th.List[int],
        hidden_features_per_covariate: th.List[th.List[int]],
        bias: bool = True,
        activation: th.Optional[str] = "torch.nn.ReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = True,
        batch_norm_args: th.Optional[dict] = None,
        n_sinkhorn_iterations: int = 10,
        tau: float = 0.1,  # used as initial temperature for sinkhorn
        tau_scheduler: th.Optional[dy.FunctionDescriptor] = None,
        # criterion
        criterion_args: th.Optional[dict] = None,
        # optimization configs [is_active(training_module, optimizer_idx) -> bool]
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
        _criterion_args = dict(terms=["ocd.training.terms.OrderedLikelihoodTerm"])
        _criterion_args.update(criterion_args or {})
        super().__init__(
            model_cls="ocd.models.order_discovery.SinkhornOrderDiscovery",
            model_args=dict(
                in_covariate_features=in_covariate_features,
                hidden_features_per_covariate=hidden_features_per_covariate,
                bias=bias,
                activation=activation,
                activation_args=activation_args,
                batch_norm=batch_norm,
                batch_norm_args=batch_norm_args,
                n_iter=n_sinkhorn_iterations,
                tau=tau if isinstance(tau, float) else 0.1,
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
        # set tau scheduler
        self.__tau_scheduler = tau_scheduler  # processed in self.tau_scheduler

    @functools.cached_property
    def tau_scheduler(self):
        if self.__tau_scheduler:
            return dy.eval_function(self.__tau_scheduler, function_of_interest="tau", dynamic_args=True)
        return dy.dynamic_args_wrapper(lambda: self.model.tau)

    def process_batch(
        self,
        batch,
        transformed_batch: th.Optional[th.Any] = None,
        transform_batch: bool = True,
    ):
        # batch would be a batch_size * num_covariates matrix
        # transformed batch should be one hot encoded for each covariate
        # for binary covariates it should be batch_size * (num_covariates * 2)

        # the number of classes for each covariate is set in
        # self.model.in_covariate_features and the number of covariates is the length of this list

        if not transform_batch:
            return transformed_batch if transformed_batch is not None else batch

        interventional_batch_size = batch[1].shape[0] if isinstance(batch, (list, tuple)) else 0

        if isinstance(batch, (list, tuple)):
            # concat the batches in the list in the first dimension
            batch = torch.cat(batch, dim=0)

        # transform batch
        # apply one hot encoding to each covariate
        # batch_size * num_covariates -> batchsize * sum(num_classes_per_covariate)
        num_classes_per_covariate = self.model.in_covariate_features
        batch_size = batch.shape[0]
        num_covariates = len(num_classes_per_covariate)
        num_classes = sum(num_classes_per_covariate)
        transformed_batch = torch.zeros(batch_size, num_classes, device=batch.device)
        start_idx = 0
        for i in range(num_covariates):
            end_idx = start_idx + num_classes_per_covariate[i]
            transformed_batch[:, start_idx:end_idx] = torch.nn.functional.one_hot(
                batch[:, i], num_classes=num_classes_per_covariate[i]
            )
            start_idx = end_idx

        if interventional_batch_size > 0:
            # split the batch into observational and interventional
            transformed_batch = [
                transformed_batch[: batch_size - interventional_batch_size],
                transformed_batch[batch_size - interventional_batch_size :],
            ]

        return transformed_batch

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
        # anneal tau
        self.model.set_tau(self.tau_scheduler(training_module=self))
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
            original_batch=batch,  # to be used by the log likelihood (todo: clean this up)
            **kwargs,
        )

    def on_train_epoch_end(self) -> None:
        # check the predicted graph structure
        # get the predicted graph structure
        permutation = sample_permutation_matrix(self.model.Gamma, n_samples=1, train=False)[0].argmax(-1)
        # compare to the true graph structure
        # get datamodule
        print(count_backward(permutation, self.trainer.datamodule.datasets[0].dag["adjmat"].values))
        self.log(
            "metrics/backwards_count",
            count_backward(permutation, self.trainer.datamodule.datasets[0].dag["adjmat"].values),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return super().on_train_epoch_end()
