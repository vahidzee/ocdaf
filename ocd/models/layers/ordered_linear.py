import torch
import typing as th


class OrderedLinear(torch.nn.Linear):
    """
    This class inherits from torch.nn.Linear because it simulates a masked Linear layer.

    Attributes:

        auto_connection: Whether to allow equal label connections.
        mask:
            A zero-one 2D mask [OUTPUT x INPUT] matrix defining the connectivity
            of output and input neurons.

    """

    def __init__(
        self,
        in_cov_features: th.List[int],
        out_cov_features: th.List[int],
        auto_connection: bool = True,
        bias: bool = True,
        dtype: th.Optional[torch.dtype] = None,
        device: th.Optional[torch.device] = None,
        mask_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initiates ordering related buffers.

        Args:
            in_cov_features: Number of input features per covariate.
            out_cov_features: Number of output features per covariate.
            auto_connection:
                A boolean value specifying whether output neurons with the
                same labels as the input neurons can be connected.
            bias: A boolean value specifying whether to add bias to the output.
            dtype: Data type of the weights.
            device: The device to instanciate the buffers on.
            masked_dtype: Data type for mask matrix.
        Retures:
            None
        """
        super().__init__(
            in_features=sum(in_cov_features),
            out_features=sum(out_cov_features),
            bias=bias,
            device=device,
            dtype=dtype,
        )

        n = len(in_cov_features)
        if n != len(out_cov_features):
            raise ValueError(
                "in_cov_features and out_cov_features must have the same length")

        dependencies = torch.tril(torch.ones(
            n, n, device=device, dtype=mask_dtype))
        if not auto_connection:
            dependencies = dependencies - \
                torch.eye(n, device=device, dtype=mask_dtype)

        self.register_buffer(
            "default_mask",
            dependencies,
        )

        self.in_cov_features = torch.tensor(in_cov_features)
        self.out_cov_features = torch.tensor(out_cov_features)

    def get_masked_weight(self, P) -> torch.Tensor:
        """

        """
        mask = P @ self.default_mask @ P.T
        mask = torch.repeat_interleave(mask, self.in_cov_features, dim=1)
        mask = torch.repeat_interleave(mask, self.out_cov_features, dim=0)
        return mask * self.weight

    def get_masked_weights(self, P: torch.Tensor) -> torch.Tensor:
        """
        Returns masked weights.
        Args:
            P: Permutation matrices [Batch_size, len(in_cov_features) x len(out_cov_features)]
        Returns:
            A `torch.Tensor` which equals to `self.weight * self.mask`.
        """
        # Permute the default mask using the permutation matrix
        # permute the mask using the permutation matrix
        batch_size = P.size(0)
        P_inv = P.transpose(1, 2)
        mask = P @ self.default_mask.unsqueeze(
            0).repeat(batch_size, 1, 1) @ P_inv
        # repeat mask[i,j] in_cov_features[i] times along the first dimension and out_cov_features[j] times along the 0th dimension
        mask = torch.repeat_interleave(mask, self.in_cov_features, dim=2)
        mask = torch.repeat_interleave(mask, self.out_cov_features, dim=1)
        return mask * self.weight

    def forward(self, inputs: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """
        Computes masked linear operation.
        Args:
            inputs: Input tensor [batch_size x [sum of [in_feautures per covariate]]]
            P: Permutation matrices [batch_size x len(in_cov_features) x len(out_cov_features)]
        Returns:
            A `torch.Tensor` which equals to masked linear operation on inputs, or:
                `inputs @ (mask * weights).T + bias` plus the input P
        """
        perm_size = len(P.shape)
        if perm_size == 3:
            masked_weights = self.get_masked_weights(P)
            ret = torch.bmm(masked_weights, inputs[:, :, None]).squeeze()
            if self.bias is not None:
                ret = ret + self.bias
        elif perm_size == 2:
            masked_weight = self.get_masked_weight(P)
            ret = torch.nn.functional.linear(inputs, masked_weight, self.bias)
        return ret, P
