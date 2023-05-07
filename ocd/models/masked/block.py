import typing as th
import torch
from .linear import MaskedLinear
import dypy


class MaskedBlock(MaskedLinear):
    def __init__(
        self,
        # linear args
        in_features: th.Union[th.List[int], int],
        out_features: th.Union[th.List[int], int],
        bias: bool = True,
        # residual
        residual: bool = False,
        # activation
        activation: th.Optional[str] = None,
        activation_args: th.Optional[dict] = None,
        # batch norm
        batch_norm: bool = False,
        batch_norm_args: th.Optional[dict] = None,
        # dropout
        dropout: th.Optional[th.Union[float, int]] = 0.0,
        # ordering args
        auto_connection: bool = True,
        reversed_ordering: bool = False,
        # general parameters
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        mask_dtype: torch.dtype = torch.uint8,
    ):
        # init linear
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            auto_connection=auto_connection,
            reversed_ordering=reversed_ordering,
            device=device,
            dtype=dtype,
            mask_dtype=mask_dtype,
        )
        self.residual = residual
        self.dropout = torch.nn.Dropout(p=dropout) if dropout else None

        self.batch_norm = (
            torch.nn.BatchNorm1d(num_features=out_features, dtype=dtype, device=device, **(batch_norm_args or dict()))
            if batch_norm
            else None
        )
        self.activation = dypy.eval(activation)(**(activation_args or dict())) if activation else None
        # todo: add more explicit control over residual
        # assert not residual or (
        #     self.in_blocks == self.out_blocks
        # ), "Residual connections are only possible if in_blocks == out_blocks"

    def forward(self, inputs: torch.Tensor, perm_mat: torch.Tensor) -> torch.Tensor:
        outputs = super().forward(inputs, perm_mat=perm_mat)
        outputs = self.batch_norm(outputs) if self.batch_norm else outputs
        outputs = self.activation(outputs) if self.activation else outputs
        outputs = self.dropout(outputs) if self.dropout else outputs
        if self.residual and self.in_blocks == self.out_blocks:
            # only perform residual connection if in_blocks == out_blocks
            outputs = outputs + inputs
        return outputs

    def extra_repr(self) -> str:
        return super().extra_repr() + f"{', reversed' if self.reversed_ordering else ''}"
