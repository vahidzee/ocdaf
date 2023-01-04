import typing as th
import torch
import numpy as np
from ocd.models.masked import MaskedMLP


class MaskedAffineFlow(MaskedMLP):
    def __init__(
        self,
        # architecture args
        in_features: th.Union[th.List[int], int],
        layers: th.List[th.Union[th.List[int], int]] = None,
        residual: bool = False,
        elementwise_perm: bool = False,
        bias: bool = True,
        activation: th.Optional[str] = "torch.nn.LeakyReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = False,
        batch_norm_args: th.Optional[dict] = None,
        # transform args
        additive: bool = False,
        # general args
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ):
        self.additive = additive
        out_features = in_features
        if not additive:
            out_features = in_features * 2 if isinstance(in_features, int) else [f * 2 for f in in_features]
        super().__init__(
            in_features=in_features,
            layers=layers,
            out_features=out_features,
            elementwise_perm=elementwise_perm,
            residual=residual,
            bias=bias,
            activation=activation,
            activation_args=activation_args,
            batch_norm=batch_norm,
            batch_norm_args=batch_norm_args,
            device=device,
            dtype=dtype,
        )

    def forward(self, inputs, **kwargs):
        autoregressive_params = super().forward(inputs, **kwargs)
        s, t = self._unconstrained_scale_and_shift(autoregressive_params)
        inputs = inputs.reshape(*inputs.shape[:-1], 1, inputs.shape[-1]) if inputs.ndim == s.ndim - 1 else inputs
        outputs = inputs * torch.exp(s) + t
        logabsdet = torch.sum(torch.abs(s), dim=-1)
        return outputs, logabsdet

    def inverse(
        self, inputs, perm_mat: th.Optional[bool] = None, elementwise_perm: th.Optional[bool] = None, **kwargs
    ):
        elementwise_perm: bool = self.elementwise_perm if elementwise_perm is None else elementwise_perm
        is_perm_batch: bool = perm_mat is not None and perm_mat.ndim == 3
        if not elementwise_perm and is_perm_batch:
            perm_mat = perm_mat.repeat_interleave(inputs.numel() // perm_mat.shape[0] // inputs.shape[-1], dim=0)
        inputs = inputs.reshape(-1, inputs.shape[-1])
        outputs = torch.zeros_like(inputs)
        for _ in range(inputs.shape[-1]):
            autoregressive_params = super().forward(outputs, perm_mat=perm_mat, elementwise_perm=True, **kwargs)
            s, t = self._unconstrained_scale_and_shift(autoregressive_params)
            outputs = (inputs - t) / torch.exp(s)
            logabsdet = -torch.sum(torch.abs(s), dim=-1)

        return outputs.unflatten(0, inputs.shape[:-1]), logabsdet.unflatten(0, inputs.shape[:-1])

    def _unconstrained_scale_and_shift(self, ar_params):
        """
        Split the autoregressive parameters into scale (s) and shift (t).
        If additive is True, then s = 0 and t = autoregressive_params.

        Returns:
            s, t (torch.Tensor): where s could be 0 if additive is True.
        """
        return (
            (torch.zeros_like(ar_params), ar_params) if self.additive else (ar_params[..., 0::2], ar_params[..., 1::2])
        )
