from typing import List, Optional, Tuple, Callable
import torch
from ocd.models.masked import MaskedMLP
import torch.nn as nn


class MaskedAffineFlowTransform(torch.nn.Module):
    def _init_stabilizing_parameters(self):
        """
        This function stabilizes the scaling parameters by ensuring that they do not explode.
        To do so, the scale parameter is always ensured to be between an interval l_\theta and r_\theta
        where l_\theta is a learnable parameter and r_\theta = l_\theta + softplus(\Delta_\theta)
        these parameters have a gradient clipping hook to ensure that they do not explode.

        s is then passed through a tanh function to ensure that it is within this interval.
        """
        self.register_parameter(
            "scaling_stability_threshold_l",
            nn.Parameter(torch.tensor(-2.5, dtype=torch.float32)),
        )
        self.register_parameter(
            "scaling_stability_threshold_interval_size",
            nn.Parameter(torch.tensor(5.0, dtype=torch.float32)),
        )
        self.register_parameter(
            "shift_stability_threshold_l",
            nn.Parameter(torch.tensor(-20.0, dtype=torch.float32)),
        )
        self.register_parameter(
            "shift_stability_threshold_interval_size",
            nn.Parameter(torch.tensor(40.0, dtype=torch.float32)),
        )
        # set a backward hook on scaling_stability_threshold_l and scaling_stability_threshold_r to ensure that
        # they do not explode
        self.scaling_stability_threshold_l.register_hook(
            lambda grad: torch.clamp(grad, min=-1e-3, max=1e-3)
        )
        self.scaling_stability_threshold_interval_size.register_hook(
            lambda grad: torch.clamp(grad, min=-1e-3, max=1e-3)
        )
        self.shift_stability_threshold_l.register_hook(
            lambda grad: torch.clamp(grad, min=-1e-3, max=1e-3)
        )
        self.shift_stability_threshold_interval_size.register_hook(
            lambda grad: torch.clamp(grad, min=-1e-3, max=1e-3)
        )

    def __init__(
        self,
        # architecture
        in_features: int,
        layers: List[int],
        dropout: Optional[float],
        residual: bool,
        activation: torch.nn.Module,
        # additional flow args
        additive: bool,
        normalization: Optional[Callable[[int], torch.nn.Module]],
        # ordering
        ordering: torch.IntTensor,
        stabilize: bool = True,
    ):
        super().__init__()
        self.additive = additive

        args = dict(
            in_features=in_features,
            layers=layers,
            residual=residual,
            activation=activation,
            ordering=ordering,
            dropout=dropout,
        )
        self.stabilize = stabilize
        if stabilize:
            self._init_stabilizing_parameters()

        self.masked_mlp_shift = MaskedMLP(**args)
        self.masked_mlp_scale = MaskedMLP(
            **args
        )  # NOTE: can remove this for memory efficiency

        self.normalization = (
            normalization(in_features) if normalization is not None else None
        )
        self.ordering = ordering

    @property
    def _scale_len(self):
        return (
            torch.nn.functional.softplus(self.scaling_stability_threshold_interval_size)
            + 1e-3
        )

    @property
    def _scale_l(self):
        return self.scaling_stability_threshold_l

    @property
    def _scale_r(self):
        return self.scaling_stability_threshold_l + self._scale_len

    @property
    def _shift_len(self):
        return (
            torch.nn.functional.softplus(self.shift_stability_threshold_interval_size)
            + 1e-3
        )

    @property
    def _shift_l(self):
        return self.shift_stability_threshold_l

    @property
    def _shift_r(self):
        return self.shift_stability_threshold_l + self._shift_len

    def _get_scale_and_shift(
        self, inputs: torch.Tensor, perm_mat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.masked_mlp_shift(inputs, perm_mat=perm_mat)
        s = (
            self.masked_mlp_scale(inputs, perm_mat=perm_mat)
            if not self.additive
            else torch.zeros_like(t)
        )
        # pass s and t through a tanh function to ensure numerical stability
        if self.stabilize:
            if not self.additive:
                interval_len = self._scale_len
                s = (
                    torch.tanh((s - self._scale_l) / interval_len) * interval_len
                    + self._scale_l
                )
            interval_len = self._shift_len
            t = (
                torch.tanh((t - self._shift_l) / interval_len) * interval_len
                + self._shift_l
            )
        return s, t

    def forward(
        self, inputs: torch.Tensor, perm_mat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        $T$ is a function from data $x$ to latent $z$ of the form:
        $$T^{-1}(z_i) = x_i = e^{s_i} z_i + t_i$$
        where $s_i$ is a function of $z_{i<}$ and $t_i$ is a function of $z_{i<}$.

        Args:
            inputs (torch.Tensor): x ~ p_x(x)
        """
        s, t = self._get_scale_and_shift(inputs, perm_mat=perm_mat)
        outputs = (inputs - t) * torch.exp(-s)
        logabsdet = -torch.sum(s, dim=-1)

        if self.normalization:
            outputs, logabsdet_ = self.normalization(outputs)
            logabsdet += logabsdet_

        return outputs, logabsdet

    def inverse(
        self,
        inputs: torch.Tensor,
        perm_mat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the inverse of the affine transformation.
        $T$ is a function from data $x$ to noise $z$ of the form:
        $$T^{-1}(z_i) = x_i = e^{s_i} z_i + t_i$$

        Args:
            inputs (torch.Tensor): the inputs to the inverse function (z's) (latent variables) of shape (batch_size, D)
            perm_mat (torch.Tensor): the permutation matrices applied to the inputs that resulted in z.
                if perm_mat is None, then the identity permutation is considered
        Returns:
            A tuple containing the outputs of the inverse function and the logabsdet of the inverse function.
        """
        d = inputs.shape[-1]
        z = inputs
        logabsdet = 0
        if self.normalization:
            z, logabsdet = self.normalization.inverse(z)
        # initialize the outputs to 0 (doesn't matter what we initialize it to)
        outputs: torch.Tensor = torch.zeros_like(z)
        # passing the outputs through the autoregressive network elementwise for d times (where d is the dimensionality
        # of the input features) will result in the inverse of the affine transformation
        for _ in range(d):
            s, t = self._get_scale_and_shift(outputs, perm_mat=perm_mat)
            outputs = torch.exp(s) * z + t

        logabsdet = logabsdet + torch.sum(
            s, dim=-1
        )  # this is the inverse of the logabsdet

        return outputs, logabsdet
