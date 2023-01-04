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
        # general args
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
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
            device=device,
            dtype=dtype,
        )

        self.latent_permutaion_model = LearnablePermutation(
            num_features=in_features if isinstance(in_features, int) else len(in_features),
            device=device,
            dtype=dtype,
        )

        # setup some model parameters
        self.elementwise_perm = elementwise_perm

    def forward(
        self,
        inputs: torch.Tensor,
        # permutation
        num_samples: th.Optional[int] = None,
        elementwise_perm: th.Optional[bool] = None,
        soft: th.Optional[bool] = True,
        # return
        return_log_prob: bool = True,
        return_noise_prob: bool = False,
        return_prior: bool = False,
    ):
        elementwise_perm = (
            elementwise_perm
            if elementwise_perm is not None
            else self.carefl.flows[0].autoregressive_net.elementwise_perm
        )
        if elementwise_perm:
            num_samples = inputs.shape[0]
        # sample latent permutation
        latent_permutation, gumbel_noise = self.latent_permutaion_model(
            inputs=inputs, num_samples=num_samples, soft=soft, return_noise=True
        )
        # log prob inputs, noise_prob, prior
        log_prob = self.carefl.log_prob(inputs, perm_mat=latent_permutation, elementwise_perm=elementwise_perm)

        # return log_prob, noise_prob, prior (if requested)
        results = [log_prob] if return_log_prob else []
        if return_noise_prob:
            results.append(gumbel_log_prob(gumbel_noise))
        if return_prior:
            raise NotImplementedError("Haven't implemented prior yet.")
        return results[0] if len(results) == 1 else tuple(results)
