import normflows as nf
import torch
from .masked import MaskedAffineFlow
import dycode as dy
import typing as th


class CAREFL(torch.nn.Module):
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
        super().__init__()
        self.base_distribution = dy.get_value(base_distribution)(**(base_distribution_args or dict()))
        self.elementwise_perm = elementwise_perm
        # instantiate flows
        self.flows = torch.nn.ModuleList(
            [
                MaskedAffineFlow(
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
        )
        if ordering is not None:
            self.reorder(torch.IntTensor(ordering))

    def forward(self, inputs, perm_mat=None, elementwise_perm: th.Optional[bool] = None, **kwargs):
        z = inputs.reshape(-1, inputs.shape[-1])
        batch_size = z.shape[0]
        log_dets = 0
        elementwise_perm = elementwise_perm if elementwise_perm is not None else self.elementwise_perm
        batch_perm = perm_mat is not None and perm_mat.ndim == 3
        permutation = perm_mat
        for i, flow in enumerate(self.flows):
            z = z.reshape(-1, inputs.shape[-1])
            z, log_det = flow(
                inputs=z,
                perm_mat=permutation,
                elementwise_perm=elementwise_perm if not i else True,
                **kwargs,
            )
            if perm_mat is not None and batch_perm and not elementwise_perm:
                # todo: check if this should be repeat interleave or repeat
                permutation = perm_mat.repeat(batch_size, 1, 1)
                log_det = log_det.reshape(-1)
            log_dets += log_det
        if perm_mat is not None and batch_perm and not elementwise_perm:
            log_dets = log_dets.reshape(batch_size, perm_mat.shape[0])
            z = z.reshape(batch_size, perm_mat.shape[0], inputs.shape[-1])
        return z.unflatten(0, inputs.shape[:-1]), log_dets.unflatten(0, inputs.shape[:-1])

    def inverse(self, inputs: torch.Tensor, **kwargs) -> th.Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the inverse of the flow.

        Args:
            inputs: Batch of inputs to invert (results of a forward pass)
            **kwargs: Additional arguments (e.g. perm_mat used in forward pass or elementwise_perm)

        Returns:
            Inverted inputs and log determinant of the inverse
        """
        z, log_dets = inputs, 0  # initialize z and log_dets
        # iterate over flows in reverse order and apply inverse
        for i, flow in enumerate(reversed(self.flows)):
            z, log_det = flow.inverse(inputs=z, **kwargs)
            log_dets += log_det  # negative sign is handled in flow.inverse

        return z, log_det

    def log_prob(self, x, z=None, log_det=None, **kwargs) -> torch.Tensor:
        """Get log probability for batch

        Args:
          x: Batch of inputs
          z: Batch of latent variables (optional, otherwise computed)
        Returns:
          log probability
        """
        z, log_det = self.forward(x, **kwargs) if z is None else (z, log_det)
        flat_z = z.reshape(-1, z.shape[-1])
        log_base_prob = self.base_distribution.log_prob(flat_z)
        log_base_prob = log_base_prob.reshape(z.shape[:-1])

        return log_base_prob - log_det

    def reorder(self, ordering: th.Optional[torch.IntTensor] = None, **kwargs) -> None:
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
