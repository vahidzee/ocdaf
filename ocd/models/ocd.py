import torch
from .carefl import CAREFL
import typing as th
from .permutation import LearnablePermutation, gumbel_log_prob
import normflows as nf


class OCD(torch.nn.Module):
    def __init__(
        self,
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
        reversed_ordering: bool = True,
        learn_permutation: bool = True,
        permutation_args: th.Optional[dict] = None,
        # general args
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        # logging args
        log_permutations: bool = False,
    ) -> None:
        super().__init__()
        self.carefl = CAREFL(
            base_distribution=base_distribution,
            base_distribution_args=base_distribution_args,
            in_features=in_features,
            layers=layers,
            elementwise_perm=elementwise_perm,
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

        if learn_permutation:
            self.permutation_model = LearnablePermutation(
                num_features=in_features if isinstance(in_features, int) else len(in_features),
                device=device,
                dtype=dtype,
                **(permutation_args or dict()),
            )
        else:
            self.permutation_model = None
        # setup some model parameters
        self.elementwise_perm = elementwise_perm

        # If logging is set to true, note that
        # this might slow down training significantly
        self.log_permutations = log_permutations
        self.logged_permutations = None
        # A simple check jic
        if self.permutation_model is None and self.log_permutations:
            raise Exception("Can only log permutations if learnable permutation is set to True.")

    def _check_logging_enabled(self):
        if not self.log_permutations:
            raise Exception("Logging is not enabled. Set log_permutations to True to enable logging.")

    def clear_logged_permutations(self):
        self._check_logging_enabled()
        self.logged_permutations = None

    def get_logged_permutations(self):
        self._check_logging_enabled()
        return self.logged_permutations

    def log_new_permutations(self, latent_permutation: torch.Tensor):
        self._check_logging_enabled()
        if self.logged_permutations is None:
            self.logged_permutations = latent_permutation
        else:
            # append the new permutations to the existing ones
            self.logged_permutations = torch.cat([self.logged_permutations, latent_permutation], dim=0)

    def forward(
        self,
        inputs: torch.Tensor,
        # permutation
        num_samples: th.Optional[int] = None,
        elementwise_perm: th.Optional[bool] = None,
        soft: th.Optional[bool] = True,
        permute: bool = True,
        # return
        return_log_prob: bool = True,
        return_noise_prob: bool = False,
        return_prior: bool = False,
        # args for dynamic methods
        **kwargs
    ):
        elementwise_perm = elementwise_perm if elementwise_perm is not None else self.carefl.flows[0].elementwise_perm
        if elementwise_perm:
            num_samples = inputs.shape[0]
        # sample latent permutation
        latent_permutation, gumbel_noise = None, None
        if self.permutation_model is not None and permute:
            latent_permutation, gumbel_noise = self.permutation_model(
                inputs=inputs, num_samples=num_samples, soft=soft, return_noise=True, **kwargs
            )

        # log the permutations that were generated if logging is enabled
        if self.log_permutations:
            self.log_new_permutations(latent_permutation)

        # log prob inputs, noise_prob, prior
        log_prob = self.carefl.log_prob(inputs, perm_mat=latent_permutation, elementwise_perm=elementwise_perm)

        # return log_prob, noise_prob, prior (if requested)
        results = [log_prob] if return_log_prob else []
        if return_noise_prob:
            results.append(gumbel_log_prob(gumbel_noise) if gumbel_noise is not None else 0)
        if return_prior:
            raise NotImplementedError("Haven't implemented prior yet.")
        return results[0] if len(results) == 1 else tuple(results)
