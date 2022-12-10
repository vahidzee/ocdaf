import typing as th
import torch
from lightning_toolbox import TrainingModule
import functools
import dycode as dy

from ocd.evaluation import backward_score, count_backward
from ocd.permutation_tools import sample_permutation, derive_deterministic_permutation


class OrderedTrainingModule(TrainingModule):
    def __init__(
        self,
        # model
        in_covariate_features: th.List[int],
        hidden_features_per_covariate: th.List[th.List[int]],
        # Fix for a permutation just to check if it works at all
        fixed_permutation: th.Optional[th.Union[th.List[int],
                                                torch.Tensor]] = None,
        # data transform
        embedding_dim: int = 0,  # 0 for one-hot encoding
        # None for no normalization,
        embedding_normalization: th.Optional[th.Union[int, str]] = None,
        embedding_normalization_eps: float = 0,
        # architecture
        bias: bool = False,
        activation: th.Optional[str] = "torch.nn.ReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = False,
        batch_norm_args: th.Optional[dict] = None,
        n_sinkhorn_iterations: int = 10,
        tau: float = 0.1,  # used as initial temperature for sinkhorn
        tau_scheduler: th.Optional[dy.FunctionDescriptor] = None,
        n_sinkhorn_scheduler: th.Optional[dy.FunctionDescriptor] = None,
        # criterion
        criterion_args: th.Optional[dict] = None,
        # optimization configs [is_active(training_module, optimizer_idx) -> bool]
        optimizer: th.Union[str, th.List[str]] = "torch.optim.Adam",
        optimizer_is_active: th.Optional[th.Union[dy.FunctionDescriptor,
                                                  th.List[dy.FunctionDescriptor]]] = None,
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
        # input stats
        self.in_covariate_features = in_covariate_features
        # initialize model and optimizer/scheduler configs
        _criterion_args = dict(
            terms=["ocd.training.terms.OrderedLikelihoodTerm"])
        _criterion_args.update(criterion_args or {})
        super().__init__(
            model_cls="ocd.models.order_discovery.SinkhornOrderDiscovery",
            model_args=dict(
                in_covariate_features=(
                    in_covariate_features if not embedding_dim else [
                        embedding_dim] * len(in_covariate_features)
                ),
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

        # set fixed permutation
        self.fixed_permutation = fixed_permutation
        if fixed_permutation is not None:
            self.fixed_permutation = fixed_permutation.copy()
            self.model.set_permutation(self.fixed_permutation)

        # embedding
        self.features_embedding = (
            torch.nn.Embedding(
                num_embeddings=len(in_covariate_features),
                embedding_dim=embedding_dim,
            )
            if embedding_dim
            else None
        )
        self.embeddings = (
            torch.nn.Embedding(
                num_embeddings=sum(in_covariate_features),
                embedding_dim=embedding_dim,
            )
            if embedding_dim
            else None
        )
        self.embedding_normalization = embedding_normalization
        self.embedding_normalization_eps = embedding_normalization_eps
        # register cummulative feature counts (for efficient embedding)
        self.register_buffer("features_cumsum", torch.cumsum(
            torch.tensor([0] + in_covariate_features[:-1]), dim=0))

        # set tau scheduler
        self.__tau_scheduler = tau_scheduler  # processed in self.tau_scheduler
        # processed in self.n_sinkhorn_scheduler
        self.__n_sinkhorn_scheduler = n_sinkhorn_scheduler

    @functools.cached_property
    def n_sinkhorn_scheduler(self):
        if self.__n_sinkhorn_scheduler:
            return dy.eval_function(self.__n_sinkhorn_scheduler, function_of_interest="n_iter", dynamic_args=True)
        return dy.dynamic_args_wrapper(lambda: self.model.n_iter)

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
        if transform_batch:
            self.temp_batch = batch
        # batch would be a batch_size * num_covariates matrix
        # transformed batch should be one hot encoded for each covariate
        # for binary covariates it should be batch_size * (num_covariates * 2)

        # the number of classes for each covariate is set in
        # self.in_covariate_features and the number of covariates is the length of this list

        if not transform_batch:
            return transformed_batch if transformed_batch is not None else batch

        batch_sizes = (
            [batch[i].shape[0] for i in range(len(batch))] if isinstance(
                batch, (list, tuple)) else [batch.shape[0]]
        )

        if isinstance(batch, (list, tuple)):
            # concat the batches in the list in the first dimension
            batch = torch.cat(batch, dim=0)

        # compute batch stats (sizes and dimensions)
        num_classes_per_covariate = self.in_covariate_features
        batch_size = batch.shape[0]
        num_covariates = len(num_classes_per_covariate)
        num_classes = sum(num_classes_per_covariate)

        # compute the embeddings (one-hot or linear)
        if self.features_embedding is not None:
            # linearly embed the covariates each covariate embedding is = value embedding  + feature embedding
            # where we have the same number of features as covariates, and the same number of values as the sum of the number of classes per covariate
            features_embeddings = self.features_embedding(
                torch.arange(len(self.in_covariate_features),
                             device=batch.device)
            )
            value_embeddings = self.embeddings(batch + self.features_cumsum)
            # normalize the embeddings
            if self.embedding_normalization is not None:
                features_embeddings = features_embeddings / (
                    features_embeddings.norm(
                        p=self.embedding_normalization, dim=1, keepdim=True)
                    + self.embedding_normalization_eps
                )
                value_embeddings = value_embeddings / (
                    value_embeddings.norm(
                        p=self.embedding_normalization, dim=1, keepdim=True)
                    + self.embedding_normalization_eps
                )
            transformed_batch = (value_embeddings +
                                 features_embeddings).reshape(batch_size, -1)
        else:
            # apply one hot encoding to each covariate
            # batch_size * num_covariates -> batchsize * sum(num_classes_per_covariate)
            transformed_batch = torch.zeros(
                batch_size, num_classes, device=batch.device)
            start_idx = 0
            for i in range(num_covariates):
                end_idx = start_idx + num_classes_per_covariate[i]
                transformed_batch[:, start_idx:end_idx] = torch.nn.functional.one_hot(
                    batch[:, i], num_classes=num_classes_per_covariate[i]
                )
                start_idx = end_idx

        # split the batch back into the original list
        transformed_batch = torch.split(transformed_batch, batch_sizes) if len(
            batch_sizes) > 1 else transformed_batch

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
        # anneal tau and change the number of sinkhorn iterations accordingly
        self.model.set_tau(self.tau_scheduler(training_module=self))
        self.model.set_n_iter(self.n_sinkhorn_scheduler(training_module=self))

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
            # to be used by the log likelihood (todo: clean this up)
            original_batch=batch,
            **kwargs,
        )

    def on_train_epoch_start(self) -> None:
        # check the predicted graph structure
        # get the predicted graph structure
        with torch.no_grad():
            permutation = self.model.get_permutation()
            # compare to the true graph structure
            # get datamodule
            self.log("metrics/tau", self.model.tau,
                     on_step=False, on_epoch=True)
            self.log("metrics/n_iter", self.model.n_iter,
                     on_step=False, on_epoch=True)
            self.log(
                "metrics/backwards_count",
                count_backward(
                    permutation, self.trainer.datamodule.datasets[0].dag),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "metrics/normalized_backwards_count",
                backward_score(
                    permutation, self.trainer.datamodule.datasets[0].dag),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return super().on_train_epoch_start()

    def get_ordering(self):
        # get the predicted causal ordering
        return self.model.get_permutation()
