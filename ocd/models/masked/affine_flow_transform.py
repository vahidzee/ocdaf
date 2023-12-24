from typing import List, Optional, Tuple
import torch
from ocd.models.masked import MaskedMLP


class MaskedAffineFlowTransform(torch.nn.Module):
    def __init__(
        self,
        # architecture
        in_features: int,
        layers: List[int],
        residual: bool,
        activation: torch.nn.Module,
        # additional flow args
        additive: bool,
        normalization: Optional[torch.nn.Module],
        # ordering
        ordering: torch.IntTensor,
    ):
        super().__init__()
        self.additive = additive

        args = dict(
            in_features=in_features,
            layers=layers,
            residual=residual,
            activation=activation,
            ordering=ordering,
        )

        self.masked_mlp_shift = MaskedMLP(**args)
        self.masked_mlp_scale = MaskedMLP(**args) if not additive else None


        self.normalization = normalization
        self.ordering = ordering


    def get_scale_and_shift(self, inputs: torch.Tensor, perm_mat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        s: torch.Tensor = self.masked_mlp_shift(inputs, perm_mat=perm_mat)
        t: torch.Tensor = self.masked_mlp_scale(inputs, perm_mat=perm_mat) if not self.additive else torch.zeros_like(s)
        if self.scale_transform_s is not None:
            s = self.scale_transform_s(s) if not self.additive else s
        if self.scale_transform_t is not None:
            t = self.scale_transform_t(t)
        return s, t

    def _get_scale_and_shift(self, inputs: torch.Tensor, perm_mat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.masked_mlp_scale(inputs, perm_mat=perm_mat)
        t = self.masked_mlp_shift(inputs, perm_mat=perm_mat) if not self.additive else torch.zeros_like(s)
        return s, t

    def forward(self, inputs: torch.Tensor, perm_mat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # initialize the outputs to 0 (doesn't matter what we initialize it to)
        outputs: torch.Tensor = torch.zeros_like(z)
        # passing the outputs through the autoregressive network elementwise for d times (where d is the dimensionality
        # of the input features) will result in the inverse of the affine transformation
        for _ in range(d):
            s, t = self._get_scale_and_shift(outputs, perm_mat=perm_mat)
            outputs = torch.exp(s) * z + t

        logabsdet = torch.sum(s, dim=-1)  # this is the inverse of the logabsdet

        return outputs, logabsdet
