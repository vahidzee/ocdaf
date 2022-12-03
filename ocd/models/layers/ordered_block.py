import typing as th
import torch
from .ordered_linear import OrderedLinear


class OrderedBlock(torch.nn.Module):

    """
    OrderedBlock is a block of ordered linear layers with optional activation and batch norm.
    Therefore, it accepts the same arguments as OrderedLinear, but also has the following arguments:

    # activation
    activation function to be applied after the linear layer. If None, no activation is applied.
    activation_args: dictionary containing the input parameters to the activation

    # batch norm
    batch_norm: boolean indicating whether batch norm should be applied after the linear layer
    batch_norm_args: dictionary containing the input parameters to the batch norm layer
    """

    def __init__(
        self,
        # linear args
        in_cov_features: th.List[int],
        out_cov_features: th.List[int],
        # bias in linears
        bias: bool = True,
        # activation
        activation=None,
        activation_args: th.Optional[dict] = None,
        # batch norm
        batch_norm: bool = True,
        batch_norm_args: th.Optional[dict] = None,
        # ordering args
        auto_connection: bool = True,
        # general parameters
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.linear = OrderedLinear(
            in_cov_features=in_cov_features,
            out_cov_features=out_cov_features,
            bias=bias,
            auto_connection=auto_connection,
            device=device,
            dtype=dtype,
        )
        # self.activation = get_value(activation)(**(activation_args or dict())) if activation else None
        self.activation = activation(**(activation_args or dict())) if activation else None
        self.batch_norm = (
            torch.nn.BatchNorm1d(
                num_features=sum(out_cov_features), dtype=dtype, device=device, **(batch_norm_args or dict())
            )
            if batch_norm
            else None
        )

    def forward(self, x):
        """
        Same functionality as OrderedLinear, but with optional activation and batch norm.
        """
        inputs, P = x
        outputs, P = self.linear(inputs, P)
        outputs = self.activation(outputs) if self.activation else outputs
        outputs = self.batch_norm(outputs) if self.batch_norm else outputs
        return (outputs, P)
