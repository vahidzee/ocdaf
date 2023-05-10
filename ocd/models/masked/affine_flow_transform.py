import typing as th
import torch
import numpy as np
from ocd.models.masked import MaskedMLP
import dypy.wrappers as dyw
import dypy as dy


@dyw.dynamize
class ScaleTransform(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        normalization: th.Optional[str] = None,
        normalization_args: th.Optional[dict] = None,
        activation: th.Optional[str] = "torch.nn.Tanh",
        activation_args: th.Optional[dict] = None,
        pre_act_shift: float = 0.0,
        pre_act_scale: float = 1./50,
        post_act_scale: float = 50.,
        post_act_shift: float = 0.0,
    ):
        super().__init__()
        self.pre_act_scale, self.post_act_scale = pre_act_scale, post_act_scale
        self.pre_act_shift, self.post_act_shift = pre_act_shift, post_act_shift
        self.normalization = (
            dy.get_value(normalization)(in_features, **(normalization_args or {}))
            if normalization is not None
            else None
        )
        activation = eval(activation) if activation is not None else activation
        self.activation = activation(**(activation_args or {})) if isinstance(activation, type) else activation

    @dyw.method
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        if self.normalization is not None:
            outputs = self.normalization(outputs)
        outputs = outputs * self.pre_act_scale + self.pre_act_shift
        outputs = self.activation(outputs) if self.activation is not None else outputs
        outputs = outputs * self.post_act_scale + self.post_act_shift
        return outputs


class MaskedAffineFlowTransform(torch.nn.Module):
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
        scale_transform: bool = True,
        scale_transform_args: th.Optional[dict] = None,
        # transform args
        additive: bool = False,
        share_parameters: bool = False,  # share parameters between scale and shift
        # ordering
        reversed_ordering: bool = False,
        # general args
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.additive: bool = additive
        self.share_parameters: bool = share_parameters

        args = dict(
            in_features=in_features,
            layers=layers,
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

        if not share_parameters:
            self.masked_mlp_shift = MaskedMLP(**args, out_features=in_features)
            self.masked_mlp_scale = MaskedMLP(**args, out_features=in_features) if not additive else None
        else:
            self.masked_mlp = MaskedMLP(
                **args, out_features=in_features * 2 if isinstance(in_features, int) else [f * 2 for f in in_features]
            )

        self.scale_transform = None
        if scale_transform is not None:
            self.scale_transform = ScaleTransform(in_features, **(scale_transform_args or {}))

    def reorder(
        self,
        ordering: th.Optional[torch.IntTensor] = None,
        seed: th.Optional[int] = None,
        mask_index: th.Optional[int] = None,
        initialization: bool = False,
    ) -> None:
        if not self.share_parameters:
            self.masked_mlp_shift.reorder(ordering, seed, mask_index, initialization)
            self.masked_mlp_scale.reorder(ordering, seed, mask_index, initialization)
        else:
            self.masked_mlp.reorder(ordering, seed, mask_index, initialization)

    def get_scale_and_shift(self, inputs: torch.Tensor, **kwargs) -> th.Tuple[torch.Tensor, torch.Tensor]:
        if self.share_parameters:
            params: th.Tuple[torch.Tensor, torch.Tensor] = self.masked_mlp(inputs, **kwargs)
            s, t = (torch.zeros_like(params), params) if self.additive else (params[..., 0::2], params[..., 1::2])
        else:
            s: torch.Tensor = self.masked_mlp_shift(inputs, **kwargs)
            t: torch.Tensor = self.masked_mlp_scale(inputs, **kwargs) if not self.additive else torch.zeros_like(s)
        if self.scale_transform is not None:
            s = self.scale_transform(s) if not self.additive else s
        return s, t

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
        s, t = self.get_scale_and_shift(inputs, **kwargs)
        outputs = (inputs - t) * torch.exp(-s)
        logabsdet = -torch.sum(s, dim=-1)
        return outputs, logabsdet

    def inverse(
        self,
        inputs: torch.Tensor,
        perm_mat: th.Optional[bool] = None,
        **kwargs,
    ) -> th.Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the inverse of the affine transformation. $T$ is a function from latent $z$ to data $x$ of the form:
        $$T(z_i) = x_i = e^{s_i} z_i + t_i$$

        Args:
            inputs (torch.Tensor): the inputs to the inverse function (z's) (output of the forward function)
            perm_mat (torch.Tensor): the permutation matrices applied to the inputs that resulted in z.
                if perm_mat is None, then no permutation matrices were applied to the inputs.
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
            s, t = self.get_scale_and_shift(outputs, perm_mat=perm_mat, **kwargs)
            outputs = torch.exp(s) * z + t
            logabsdet = torch.sum(s, dim=-1)  # this is the inverse of the logabsdet

        # unflatten the outputs and logabsdet to match the original batch shape
        return outputs.unflatten(0, inputs.shape[:-1]), logabsdet.unflatten(0, inputs.shape[:-1])

    def extra_repr(self):
        additive = f", additive={self.additive}" if self.additive else ""
        ordering = (
            f"ordering={self.masked_mlp.ordering}"
            if self.share_parameters
            else f"ordering={self.masked_mlp_shift.ordering}"
        )
        return super().extra_repr() + ordering + additive
