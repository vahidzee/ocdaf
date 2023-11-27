from typing import Optional, List, Tuple, Type
import torch


class MaskedLinear(torch.nn.Linear):
    def __init__(
        self,
        dims: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        self_connection: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)
        self.auto_connection: bool = self_connection
        self.dims: int = dims

        # setup a cannonical mask
        _ordering = torch.arange(dims).to(device=device, dtype=dtype)
        _connection_criteria = torch.less_equal if self.auto_connection else torch.less
        self.register_buffer(
            "mask",
            (_connection_criteria(_ordering[:, None], _ordering[None, :]).T).to(dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor, p: Optional[torch.Tensor]) -> torch.Tensor:
        m = self.mask.to(dtype=x.dtype)
        if p is not None:
            # batch the permutations [P, NxN]
            p = p.unsqueeze(0) if p.ndim == 2 else p
            # permute the cannonical mask
            p = p.to(device=self.weight.device, dtype=x.dtype)
            m = p @ m @ p.transpose(-2, -1)  # [P, NxN]

        # mask the weights
        m = torch.repeat_interleave(m, self.in_features // self.dims, dim=-1)
        m = torch.repeat_interleave(m, self.out_features // self.dims, dim=-2)
        w = (m * self.weight).transpose(-2, -1)  # [P, Weights]

        # process the masked linear outputs
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        w = w.repeat_interleave(x.shape[0] // w.shape[0], dim=0)
        return torch.bmm(x.unsqueeze(1), w).squeeze(1).reshape(*original_shape[:-1], self.out_features)


class MaskedMLP(torch.nn.Module):
    # TODO: do we want batchnorms? and doube check residuals
    def __init__(
        self,
        dims: int,
        layers: List[int],
        bias: bool = True,
        activation: torch.nn.Module = None,
        residual: bool = False,
        # general parameters
        self_connection: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        features = [dims] + [l * dims for l in (layers or [])]
        self.residual = residual
        self.activation = activation

        self.layers = torch.nn.ModuleList()
        for i in range(1, len(features)):
            layer = MaskedLinear(
                dims=dims,
                in_features=features[i - 1],
                out_features=features[i],
                bias=bias,
                self_connection=True if i < len(layers) else self_connection,
                dtype=dtype,
                device=device,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, p: Optional[torch.Tensor] = None):
        for i, layer in enumerate(self.layers):
            y = layer(x=x, p=p)
            if self.activation is not None and i != len(self.layers):
                y = self.activation(y)

            x = (x + y) if self.residual and x.shape[-1] == y.shape[-1] else y
        return x


class AffineTransform(torch.nn.Module):
    def __init__(
        self,
        dims: int,
        layers: List[int],
        bias: bool = True,
        residual: bool = False,
        activation: Optional[torch.nn.Module] = torch.nn.LeakyReLU(),
        normalization: Optional[Type[torch.nn.Module]] = None,
        additive: bool = False,
        share_parameters: bool = False,  # share parameters between scale and shift
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.additive: bool = additive
        self.share_parameters: bool = share_parameters

        args = dict(
            dims=dims,
            residual=residual,
            layers=(layers or []) + [1 if share_parameters else (2 if additive else 1)],
            bias=bias,
            activation=activation,
            auto_connection=False,
            device=device,
            dtype=dtype,
        )

        if not share_parameters:
            self.shift = MaskedMLP(**args)
            self.scale = MaskedMLP(**args) if not additive else None
        else:
            self.scale_shift = MaskedMLP(**args)
        # TODO: setup normalization
        self.normalization = normalization

    def get_scale_shift(self, x: torch.Tensor, p: Optional[torch.Tensor] = None):
        if self.share_parameters:
            params = self.scale_shift(x=x, p=p)
            s, t = (torch.zeros_like(params), params) if self.additive else (params[..., 0::2], params[..., 1::2])
        else:
            s = self.scale(x, p=p)
            t = self.shift(x, p=p) if not self.additive else torch.zeros_like(s)
        # TODO: add normalization here
        return s, t

    def forward(self, x: torch.Tensor, p: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        s, t = self.get_scale_shift(x=x, p=p)
        outputs = (x - t) * torch.exp(-s)
        logabsdet = -torch.sum(s, dim=-1)
        return outputs, logabsdet

    def inverse(self, x: torch.Tensor, p: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        for _ in range(x.shape[-1]):
            s, t = self.get_scale_and_shift(x=x, p=p)
            x = torch.exp(s) * x + t
            logabsdet = torch.sum(s, dim=-1)
        return x, logabsdet


class OSlow(torch.nn.Module):
    def __init__(
        self,
        dims: int,
        layers: List[int],
        num_transforms: int = 1,
        bias: bool = True,
        residual: bool = False,
        activation: Optional[torch.nn.Module] = torch.nn.LeakyReLU(),
        normalization: Optional[Type[torch.nn.Module]] = None,
        additive: bool = False,
        share_parameters: bool = False,  # share parameters between scale and shift
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # base distribution
        base_distribution: torch.distributions.Distribution = torch.distributions.Normal(),
        # gamma params
        # TODO: add gamma things here
    ):
        super().__init__()
        self.transforms = torch.nn.ModuleList()
        for _ in range(num_transforms):
            transform = AffineTransform(
                dims=dims,
                layers=layers,
                bias=bias,
                residual=residual,
                normalization=normalization,
                activation=activation,
                additive=additive,
                share_parameters=share_parameters,
                device=device,
                dtype=dtype,
            )
            self.transforms.append(transform)

        self.base_distribution = base_distribution

    def forward(self, x: torch.Tensor, p: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = 0
        for transform in self.transforms:
            x, ld = transform(x=x, p=p)
            log_det += ld
        return x, log_det

    def inverse(self, x: torch.Tensor, p: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        z, log_dets = inputs, 0  # initialize z and log_det
        # iterate over flows in reverse order and apply inverse

        for i, flow in enumerate(reversed(self)):
            z, log_det = flow.inverse(inputs=z, **kwargs)
            log_dets += log_det  # sign is handled in flow.inverse
        return z, log_det

    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        # Get the device of the current model
        device = next(self[0].parameters()).device
        # Set the noises and set their device
        z = self.base_distribution.sample((num_samples, self.in_features)).to(device)

        return self.inverse(z, **kwargs)[0]

    def log_prob(self, x=None, z=None, logabsdet=None, **kwargs) -> torch.Tensor:
        assert (x is None) != (z is None), "Either x or z must be None"
        z, logabsdet = self.forward(x, **kwargs) if z is None else (z, logabsdet)
        flat_z = z.reshape(-1, z.shape[-1])
        # print the maximum and minimum values of the latent variables
        log_base_prob = self.base_distribution.log_prob(flat_z).sum(-1)
        log_base_prob = log_base_prob.reshape(z.shape[:-1])
        return log_base_prob + logabsdet

    def intervene(self, num_samples, intervention: Dict[int, dy.FunctionDescriptor], **kwargs):
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

    def do(self, idx, values: Union[torch.Tensor, list], target: Optional[int] = None, num_samples=50):
        values = values.reshape(-1).tolist() if isinstance(values, torch.Tensor) else values
        results = torch.stack([self.intervene(num_samples, {idx: value}) for value in values], dim=0)
        return results[:, :, target] if target is not None else results
