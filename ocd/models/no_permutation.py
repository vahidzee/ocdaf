import torch
import dypy
import typing as th
import functools


class MaskedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        auto_connection: bool = True,  #
        data_to_noise: bool = False,
        mask_dtype: torch.dtype = torch.uint8,
    ) -> None:
        # setup linear
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        # setup mask and ordering
        self.data_to_noise, self.auto_connection = data_to_noise, auto_connection
        self.register_buffer("ordering", torch.empty(out_features, device=device, dtype=torch.int))
        self.register_buffer("mask", torch.ones(out_features, in_features, device=device, dtype=mask_dtype))

    def reorder(self, inputs_ordering: torch.IntTensor, ordering: torch.IntTensor) -> None:
        # setup mask and ordering of this layer
        self.ordering.data.copy_(torch.repeat_interleave(ordering, self.ordering.shape[0] // ordering.shape[0]))
        self.mask.data.copy_(self.connection_operator(inputs_ordering[:, None], self.ordering[None, :]).T)

    @functools.cached_property
    def connection_operator(self):
        # used to create the mask
        if not self.data_to_noise:
            return torch.less_equal if self.auto_connection else torch.less  # noise to data
        return torch.greater_equal if self.auto_connection else torch.greater  # data to noise

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(inputs, self.mask * self.weight, self.bias)


class MaskedBlock(MaskedLinear):
    def __init__(
        self,
        residual: bool = False,
        activation: th.Optional[str] = "torch.nn.LeakyReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = False,
        batch_norm_args: th.Optional[dict] = None,
        dropout: th.Optional[th.Union[float, int]] = 0.0,
        **linear_args,
    ):
        super().__init__(**linear_args)
        self.residual = residual
        self.dropout = torch.nn.Dropout(p=dropout) if dropout else None
        self.batch_norm = None
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(
                num_features=self.out_features,
                dtype=self.mask.dtype,
                device=self.mask.device,
                **(batch_norm_args or dict()),
            )
        self.activation = dypy.eval(activation)(**(activation_args or dict())) if activation else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = super().forward(inputs)
        outputs = self.batch_norm(outputs) if self.batch_norm else outputs
        outputs = self.activation(outputs) if self.activation else outputs
        outputs = self.dropout(outputs) if self.dropout else outputs
        if self.residual and self.in_features == self.out_features:
            outputs = outputs + inputs
        return outputs


class MaskedMLP(torch.nn.ModuleList):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layers: th.List[int] = None,
        # residual
        residual: bool = False,
        # blocks
        bias: bool = True,
        activation: th.Optional[str] = "torch.nn.LeakyReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = False,
        batch_norm_args: th.Optional[dict] = None,
        # general parameters
        auto_connection: bool = False,
        data_to_noise: bool = False,
        ordering: th.Optional[list] = None,
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ):
        """
        Initialize a Masked (autoregressive) Multi-Layer Perceptron.
        """
        super().__init__()
        layers = (layers or []) + [out_features]

        # set architectural hyperparameters
        self.in_features, self.out_features = in_features, layers[-1]

        # initialize layers
        for i, layer_features in enumerate(layers):
            self.append(
                MaskedBlock(
                    in_features=in_features if not i else self[-1].out_features,
                    out_features=layer_features,
                    bias=bias,
                    data_to_noise=data_to_noise,
                    activation=activation if i < len(layers) - 1 else None,
                    auto_connection=True if i < len(layers) - 1 else auto_connection,
                    activation_args=activation_args if i < len(layers) - 1 else None,
                    batch_norm=batch_norm if i < len(layers) - 1 else False,
                    batch_norm_args=batch_norm_args if i < len(layers) - 1 else None,
                    residual=residual if i < len(layers) - 1 else False,
                )
            )

        # initialize ordering
        if ordering is None:
            ordering = torch.arange(self.in_features, dtype=torch.int, device=device)
        else:
            ordering = torch.IntTensor(ordering) if isinstance(ordering, list) else ordering
        self.register_buffer("ordering", ordering)
        self.reorder(ordering)  # initialize masks and orderings

    def reorder(self, ordering: th.Optional[torch.IntTensor] = None) -> None:
        if ordering is not None:
            self.ordering.data.copy_(ordering)
        for i, layer in enumerate(self):
            layer.reorder(inputs_ordering=self.ordering if not i else self[i - 1].ordering, ordering=self.ordering)

    def forward(self, inputs: torch.Tensor):
        for layer in self:
            inputs = layer(inputs)
        return inputs

    @property
    def orderings(self):
        return [self.ordering] + [layer.ordering for layer in self]


class MaskedAffineFlow(MaskedMLP):
    # goes from data to noise (forward)
    def __init__(self, in_features: int, additive: bool = False, **masked_mlp_args):
        self.additive = additive
        super().__init__(
            in_features=in_features,
            out_features=(in_features * 2 if not additive else in_features),
            data_to_noise=False,
            **masked_mlp_args,
        )

    def forward(self, inputs: torch.Tensor) -> th.Tuple[torch.Tensor, torch.Tensor]:
        """
        This function is $T^{-1}$ where $T$ is a function from latent $z$ to data $x$ of the form:
            $$T(z_i) = x_i = e^{s_i} z_i + t_i$$
        where $s_i$ is a function of $z_{i<}$ and $t_i$ is a function of $z_{i<}$.

        Therefore the $T^{-1}$ would be:
        $$T^{-1}(x_i) = z_i = \frac{x_i - t_i}{e^{s_i}}$$
        """
        s, t = self._split_st(super().forward(inputs))
        outputs = inputs * torch.exp(-s) - t * torch.exp(-s)
        logabsdet = -torch.sum(s, dim=-1)
        # outputs = (inputs - t) * torch.nn.functional.softplus(s)
        # logabsdet = torch.sum(torch.log(torch.nn.functional.softplus(s)), dim=-1)
        return outputs, logabsdet

    def _split_st(self, st):
        return (torch.zeros_like(st), st) if self.additive else (st[..., 0::2], st[..., 1::2])


class OCDAF(torch.nn.Module):
    def __init__(
        self,
        num_transforms: int = 1,
        ordering: th.Optional[torch.IntTensor] = None,
        **flow_args,
    ):
        super().__init__()
        # self.base_distribution = dypy.get_value(base_distribution)(**(base_distribution_args or dict()))
        # self.elementwise_perm = elementwise_perm
        # instantiate flows
        self.base_distribution = torch.distributions.Normal(0, 1)
        self.flows = torch.nn.ModuleList(
            [MaskedAffineFlow(**flow_args, ordering=ordering) for _ in range(num_transforms)]
        )

    def forward(self, inputs):
        log_dets, z = 0, inputs
        for flow in self.flows:
            z, log_det = flow(inputs=z)
            log_dets += log_det
        return z, log_dets

    def log_prob(self, z, logabsdet) -> torch.Tensor:
        # if z.requires_grad:
        # logabsdet.register_hook(lambda grad: torch.where(torch.isnan(grad) + torch.isinf(grad), torch.zeros_like(grad), grad))
        # z.register_hook(lambda grad: torch.where(torch.isnan(grad) + torch.isinf(grad), torch.zeros_like(grad), grad))
        log_base_prob = self.base_distribution.log_prob(z).sum(-1)
        return log_base_prob + logabsdet

    @property
    def orderings(self) -> torch.IntTensor:
        return [flow.orderings for flow in self.flows]
