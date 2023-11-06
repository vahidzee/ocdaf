import torch
from .masked import MaskedAffineFlowTransform
from .permutation.utils import translate_idx_ordering
import dypy as dy
import typing as th
import functools

import torch


class OSlow(torch.nn.ModuleList):
    def __init__(
        self,
        # architecture
        in_features: th.Union[th.List[int], int],
        layers: th.List[th.Union[th.List[int], int]] = None,
        residual: bool = False,
        bias: bool = True,
        activation: th.Optional[str] = "torch.nn.LeakyReLU",
        activation_args: th.Optional[dict] = None,
        # batch norm
        batch_norm: bool = False,
        batch_norm_args: th.Optional[dict] = None,
        # additional flow args
        additive: bool = False,
        share_parameters: bool = False,  # share parameters between scale and shift
        num_transforms: int = 1,
        scale_transform: bool = True,
        scale_transform_s_args: th.Optional[dict] = None,
        scale_transform_t_args: th.Optional[dict] = None,
        # base distribution
        base_distribution: th.Union[torch.distributions.Distribution, str] = "torch.distributions.Normal",
        base_distribution_args: dict = dict(loc=0.0, scale=1.0),  # type: ignore
        # ordering
        ordering: th.Optional[torch.IntTensor] = None,
        reversed_ordering: bool = False,
        # general args
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.base_distribution = dy.get_value(base_distribution)(**(base_distribution_args or dict()))
        self.in_features = in_features
        # instantiate flows
        for _ in range(num_transforms):
            self.append(
                MaskedAffineFlowTransform(
                    in_features=in_features,
                    layers=layers,
                    residual=residual,
                    bias=bias,
                    activation=activation,
                    activation_args=activation_args,
                    batch_norm=batch_norm,
                    batch_norm_args=batch_norm_args,
                    additive=additive,
                    scale_transform=scale_transform,
                    scale_transform_s_args=scale_transform_s_args,
                    scale_transform_t_args=scale_transform_t_args,
                    share_parameters=share_parameters,
                    reversed_ordering=reversed_ordering,
                    device=device,
                    dtype=dtype,
                )
            )
        if ordering is not None:
            self.reorder(torch.IntTensor(translate_idx_ordering(ordering)))

    def forward(self, inputs, perm_mat=None, return_intermediate_results: bool = False, **kwargs):
        """
        Args:
            inputs: samples from the data distribution (x's)
            perm_mat: permutation matrix to use for the flow (if None, then no permutation is used) (N, D, D)
            return_intermediate_results: whether to return the intermediate results (z's) of the flow transformations
        Returns:
            z: transformed samples from the base distribution (z's)
        """

        log_dets, z = 0, inputs.reshape(-1, inputs.shape[-1])
        results = []
        for i, flow in enumerate(self):
            z = z.reshape(-1, inputs.shape[-1])
            if return_intermediate_results:
                results.append(z)
            z, log_det = flow(inputs=z, perm_mat=perm_mat, **kwargs)
            log_dets += log_det.reshape(-1)

        z, log_dets = z.unflatten(0, inputs.shape[:-1]), log_dets.unflatten(0, inputs.shape[:-1])
        if return_intermediate_results:
            results.append(z)
            return results
        return z, log_dets


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

        for i, flow in enumerate(reversed(self)):
            z, log_det = flow.inverse(inputs=z, **kwargs)
            log_dets += log_det  # sign is handled in flow.inverse
        return z, log_det

    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        """Sample from the flow

        Args:
          num_samples: Number of samples to generate
        Returns:
          Samples
        """
        # Get the device of the current model
        device = next(self[0].parameters()).device
        # Set the noises and set their device
        z = self.base_distribution.sample((num_samples, self.in_features)).to(device)

        return self.inverse(z, **kwargs)[0]

    def log_prob(self, x=None, z=None, logabsdet=None, **kwargs) -> torch.Tensor:
        """Get log probability for batch

        $$\log p_x(x) = \log p_z(T^{-1}(x)) + \log |det(J(T^{-1}(x)))|$$

        Args:
          x: Batch of inputs
          z: Batch of latent variables (optional, otherwise computed)
        Returns:
          log probability
        """
        assert (x is None) != (z is None), "Either x or z must be None"
        z, logabsdet = self.forward(x, **kwargs) if z is None else (z, logabsdet)
        flat_z = z.reshape(-1, z.shape[-1])
        # print the maximum and minimum values of the latent variables
        log_base_prob = self.base_distribution.log_prob(flat_z).sum(-1)
        log_base_prob = log_base_prob.reshape(z.shape[:-1])
        return log_base_prob + logabsdet

    def reorder(self, ordering: th.Optional[torch.IntTensor] = None, **kwargs) -> None:
        if ordering is not None:
            ordering = torch.IntTensor(ordering)
        for flow in self:
            flow.reorder(ordering, **kwargs)

    def intervene(self, num_samples, intervention: th.Dict[int, dy.FunctionDescriptor], **kwargs):
        intervention = {k: dy.eval(f) for k, f in intervention.items()}
        device = next(self[0].parameters()).device

        # Set the noises and set their device
        base_z = self.base_distribution.sample((num_samples, self.in_features)).to(device)
        x = self.inverse(base_z, **kwargs)[0]

        for idx in intervention:
            x[:, idx] = intervention[idx](x) if callable(intervention[idx]) else intervention[idx]
            z = self(x, **kwargs)[0]
            z[:, idx + 1 :] = base_z[:, idx + 1 :]
            x = self.inverse(z, **kwargs)[0]
        return x

    def do(self, idx, values: th.Union[torch.Tensor, list], target: th.Optional[int] = None, num_samples=50):
        values = values.reshape(-1).tolist() if isinstance(values, torch.Tensor) else values
        results = torch.stack([self.intervene(num_samples, {idx: value}) for value in values], dim=0)
        return results[:, :, target] if target is not None else results

    @property
    def ordering(self) -> torch.IntTensor:
        return self[0].ordering

    @property
    def orderings(self) -> torch.IntTensor:
        return [flow.orderings for flow in self]
