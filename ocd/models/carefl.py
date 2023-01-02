import normflows as nf
import torch
from ocd.models.normflows.flow import NormalizingFlow
from ocd.models.normflows.autoregressive import MaskedAffineAutoregressive
import dycode as dy
import typing as th


class CAREFL(NormalizingFlow):
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
    ):
        flows = [
            MaskedAffineAutoregressive(
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
                device=device,
                dtype=dtype,
            )
            for _ in range(num_transforms)
        ]
        super().__init__(flows=flows, q0=dy.get_value(base_distribution)(**(base_distribution_args or dict())))
        if ordering is not None:
            self.reorder(torch.IntTensor(ordering))

    def reorder(
        self,
        ordering: th.Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> None:
        if ordering is not None:
            ordering = torch.IntTensor(ordering)
        for flow in self.flows:
            flow.reorder(ordering, **kwargs)

    @property
    def ordering(self) -> torch.IntTensor:
        return self.flows[0].ordering

    @property
    def orderings(self) -> torch.IntTensor:
        return [flow.orderings for flow in self.flows]
