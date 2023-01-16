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
        # ordering
        reversed_ordering: bool = True,
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
            auto_connection=False,
            reversed_ordering=reversed_ordering,
            device=device,
            dtype=dtype,
        )

    def compute_dependencies(
        self, inputs: th.Optional[torch.Tensor] = None, *, forward: bool = True, **kwargs
    ) -> torch.Tensor:
        return super().compute_dependencies(inputs, **kwargs, forward_function="forward" if forward else "inverse")

    def forward(self, inputs: torch.Tensor, **kwargs) -> th.Tuple[torch.Tensor, torch.Tensor]:
        """
        $T^{-1}$ is the inverse of $T$. $T$ is a function from latent $z$ to data $x$ of the form:
        $$T(z_i) = x_i = e^{s_i} z_i + t_i$$
        where $s_i$ is a function of $z_{i<}$ and $t_i$ is a function of $z_{i<}$.

        Therefore the $T^{-1}$ would be:
        $$T^{-1}(x_i) = z_i = \frac{x_i - t_i}{e^{s_i}}$$

        Args:
            inputs (torch.Tensor): x ~ p_x(x)
        """
        autoregressive_params: th.Tuple[torch.Tensor, torch.Tensor] = super().forward(inputs, **kwargs)
        s, t = self._split_scale_and_shift(autoregressive_params)
        inputs = inputs.reshape(*inputs.shape[:-1], 1, inputs.shape[-1]) if inputs.ndim == s.ndim - 1 else inputs
        outputs = inputs * torch.exp(-s) - t * torch.exp(-s)
        logabsdet = -torch.sum(s, dim=-1)
        return outputs, logabsdet

    def inverse(
        self,
        inputs: torch.Tensor,
        perm_mat: th.Optional[bool] = None,
        elementwise_perm: th.Optional[bool] = None,
        **kwargs,
    ) -> th.Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the inverse of the affine transformation. $T$ is a function from latent $z$ to data $x$ of the form:
        $$T(z_i) = x_i = e^{s_i} z_i + t_i$$

        Args:
            inputs (torch.Tensor): the inputs to the inverse function (z's) (output of the forward function)
            perm_mat (torch.Tensor): the permutation matrices applied to the inputs that resulted in z.
                if perm_mat is None, then no permutation matrices were applied to the inputs.
            elementwise_perm (bool): whether or not to apply the permutation matrices elementwise
                to each z in the batch. if elementwise_perm is None, then self.elementwise_perm is used.
                This argument is only used to override self.elementwise_perm if needed, for testing purposes.
            **kwargs: additional keyword arguments to pass to the autoregressive network. (e.g. mask_idx)
        Returns:
            A tuple containing the outputs of the inverse function and the logabsdet of the inverse function.
        """
        z: torch.Tensor = inputs.reshape(-1, inputs.shape[-1])
        # initialize the outputs to 0 (doesn't matter what we initialize it to)
        outputs: torch.Tensor = torch.zeros_like(z)
        # passing the outputs through the autoregressive network elementwise for d times (where d is the dimensionality
        # of the input features) will result in the inverse of the affine transformation
        for _ in range(inputs.shape[-1]):
            autoregressive_params = super().forward(outputs, perm_mat=perm_mat, elementwise_perm=True, **kwargs)
            s, t = self._split_scale_and_shift(autoregressive_params)
            outputs = torch.exp(s) * z + t  # this is the inverse of the affine transformation
            logabsdet = torch.sum(s, dim=-1)  # this is the inverse of the logabsdet

        # unflatten the outputs and logabsdet to match the original batch shape
        return outputs.unflatten(0, inputs.shape[:-1]), logabsdet.unflatten(0, inputs.shape[:-1])

    def _split_scale_and_shift(self, ar_params):
        """
        Split the autoregressive parameters into scale (s) and shift (t).
        If additive is True, then s = 0 and t = autoregressive_params.

        Returns:
            s, t (torch.Tensor): where s could be 0 if additive is True.
        """
        return (
            (torch.zeros_like(ar_params), ar_params) if self.additive else (ar_params[..., 0::2], ar_params[..., 1::2])
        )
