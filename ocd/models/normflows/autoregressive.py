import normflows as nf
import typing as th
import torch
import numpy as np
from ocd.models.masked import MaskedMLP


class MaskedAffineAutoregressive(nf.flows.affine.autoregressive.Autoregressive):
    def __init__(
        self,
        # architecture args
        in_features: th.Union[th.List[int], int],
        layers: th.List[th.Union[th.List[int], int]] = None,
        residual: bool = False,
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
        net = MaskedMLP(
            in_features=in_features,
            layers=layers,
            out_features=out_features,
            residual=residual,
            bias=bias,
            activation=activation,
            activation_args=activation_args,
            batch_norm=batch_norm,
            batch_norm_args=batch_norm_args,
            device=device,
            dtype=dtype,
        )
        super().__init__(autoregressive_net=net)

    def forward(self, inputs, **kwargs):
        autoregressive_params = self.autoregressive_net(inputs, **kwargs)
        s, t = self._unconstrained_scale_and_shift(autoregressive_params)
        outputs = inputs * torch.exp(s) + t
        logabsdet = nf.utils.sum_except_batch(torch.abs(s), num_batch_dims=1)
        return outputs, logabsdet

    def inverse(self, inputs, **kwargs):
        num_inputs = np.prod(inputs.shape[1:])
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(outputs, **kwargs)
            s, t = self._unconstrained_scale_and_shift(autoregressive_params)
            outputs = (inputs - t) / torch.exp(s)
            logabsdet = -nf.utils.sum_except_batch(torch.abs(s), num_batch_dims=1)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, ar_params):
        """
        Split the autoregressive parameters into scale (s) and shift (t).
        If additive is True, then s = 0 and t = autoregressive_params.

        Returns:
            s, t (torch.Tensor): where s could be 0 if additive is True.
        """
        return (torch.zeros_like(ar_params), ar_params) if self.additive else (ar_params[:, 0::2], ar_params[:, 1::2])

    def reorder(
        self,
        ordering: th.Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> None:
        return self.autoregressive_net.reorder(ordering=ordering, **kwargs)

    @property
    def ordering(self):
        return self.autoregressive_net.ordering

    @property
    def orderings(self):
        return self.autoregressive_net.orderings
