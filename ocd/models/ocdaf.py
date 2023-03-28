import torch
from ocd.models.affine_flow import AffineFlow
import typing as th
from ocd.models.permutation import LearnablePermutation, gumbel_log_prob
import dypy as dy
from lightning_toolbox import TrainingModule


class OCDAF(torch.nn.Module):
    def __init__(
        self,
        # architecture
        in_features: th.Union[th.List[int], int],
        layers: th.List[th.Union[th.List[int], int]] = None,
        residual: bool = False,
        bias: bool = True,
        activation: th.Optional[str] = "torch.nn.LeakyReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = False,
        batch_norm_args: th.Optional[dict] = None,
        # additional flow args
        additive: bool = False,
        num_transforms: int = 1,
        # base distribution
        base_distribution: th.Union[torch.distributions.Distribution, str] = "torch.distributions.Normal",
        base_distribution_args: dict = dict(loc=0.0, scale=1.0),  # type: ignore
        # ordering
        ordering: th.Optional[torch.IntTensor] = None,
        reversed_ordering: bool = False,
        use_permutation: bool = True,
        permutation_learner_cls: th.Optional[str] = "ocd.models.permutation.LearnablePermutation",
        permutation_learner_args: th.Optional[dict] = None,
        # general args
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # get an IntTensor of 1 to N
        self.flow = AffineFlow(
            base_distribution=base_distribution,
            base_distribution_args=base_distribution_args,
            in_features=in_features,
            layers=layers,
            residual=residual,
            bias=bias,
            activation=activation,
            activation_args=activation_args,
            batch_norm=batch_norm,
            batch_norm_args=batch_norm_args,
            additive=additive,
            num_transforms=num_transforms,
            ordering=ordering,
            reversed_ordering=reversed_ordering,
            device=device,
            dtype=dtype,
        )

        self.ordering = ordering

        if use_permutation:
            self.permutation_model = dy.get_value(permutation_learner_cls)(
                num_features=in_features if isinstance(in_features, int) else len(in_features),
                device=device,
                dtype=dtype,
                **(permutation_learner_args or dict()),
            )
        else:
            self.permutation_model = None

    def forward(
        self,
        inputs: torch.Tensor,
        # permutation
        soft: th.Optional[bool] = True,
        permute: bool = True,
        # return
        return_log_prob: bool = True,
        return_noise_prob: bool = False,
        return_prior: bool = False,
        return_latent_permutation: bool = False,
        # args for dynamic methods
        training_module: th.Optional[TrainingModule] = None,
        **kwargs
    ):
        num_samples = inputs.shape[0]
        # sample latent permutation
        latent_permutation, gumbel_noise = None, None
        if self.permutation_model is not None and permute:
            latent_permutation, gumbel_noise = self.permutation_model(
                inputs=inputs, num_samples=inputs.shape[0], soft=soft, return_noise=True, **kwargs
            )

        if training_module is not None:
            training_module.remember(inputs=inputs)
            training_module.remember(perm_mat=latent_permutation)
        log_prob = self.flow.log_prob(inputs, perm_mat=latent_permutation)
        if training_module is not None:
            training_module.remember(log_prob=log_prob)


        # return log_prob, noise_prob, prior (if requested)
        results = dict(log_prob=log_prob) if return_log_prob else dict()
        if return_noise_prob:
            results["noise_prob"] = gumbel_log_prob(gumbel_noise) if gumbel_noise is not None else 0
        if return_prior:
            raise NotImplementedError("Haven't implemented prior yet.")
        if return_latent_permutation:
            results["latent_permutation"] = latent_permutation
        return results
