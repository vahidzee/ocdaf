import torch
import typing as th
import functools


class MaskedLinear(torch.nn.Linear):
    """
    Linear layer with ordered outputs to maintain an autoregressive data flow.

    Two modes are supported:
        - block mode: the output neurons are are ordered in blocks. Each block can
            have a different number of neurons. The number of neurons in each block
            is determined how `out_features` is specified.

            For example, if `input_features` is [3, 4, 2, 3] and `out_features` is
            [10, 5, 7, 3], then the output neurons are ordered in blocks of size
            10, 5, 7, and 3. If a natural ordering is used, then the first 10 neurons
            will be labeled 0, and the next 5 will be labeled 1, and so on.

            For the sake of simplicity when initilizing the model, if `out_features`
            is an int and `input_features` is a list, then the output features are
            split evenly per input dimension. For example, if `input_features` is
            [3, 4, 2, 3] and `out_features` is 20, then the output neurons are
            ordered in blocks of size 20, 20, 20, and 20.

        - single mode: the output neurons are ordered in a single block.

    Attributes:
        ordering: current ordering of the output neurons.
        mask: a zero-one 2D mask matrix defining the connectivity of output and
            input neurons.
    """

    def __init__(
        self,
        in_features: th.Union[th.List[int], int],
        out_features: th.Union[th.List[int], int],
        bias: bool = True,
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        auto_connection: bool = True,
        reversed_ordering: bool = False,
        mask_dtype: torch.dtype = torch.uint8,
    ) -> None:
        """
        innitializes module by initializing an ordering mixin and a linear layer.

        Args:
            in_features:
                number of input feature dimensions in 1D. (or a list of features per input dimension)
            out_features:
                number of output feature dimensions in 1D. (or a list of features per output dimension)
                if in_features is a list and out_features is an int, then the output features are split evenly
                per input dimension.
            bias: whether to use a bias vector for the linear operation.
            device: the destination device for the buffers and parameters.
            dtype: data type for linear layer parameters.
            masked_dtype: data type for mask matrix.
            auto_connection: whether to allow equal label connections.
            masked_dtype: data type for mask matrix.

        Returns:
            None
        """
        # validate hyperparameters
        assert not (
            isinstance(in_features, int) and isinstance(out_features, list)
        ), "if in_features is an int, out_features must be an int."
        # only allow out_features to be a list of the same length as in_features or a single int
        if isinstance(in_features, list) and isinstance(out_features, list) and len(in_features) != len(out_features):
            raise ValueError(
                f"if in_features is a list, out_features must be a list of the same length. "
                f"got {len(in_features)} and {len(out_features)}."
            )

        # setup underlying linear layer
        num_in_features = in_features if isinstance(in_features, int) else sum(in_features)
        num_out_features = out_features
        if isinstance(in_features, list):
            num_out_features = sum(out_features) if isinstance(out_features, list) else out_features * len(in_features)
        assert num_out_features > 0, "number of output features must be greater than 0."
        assert num_in_features > 0, "number of input features must be greater than 0."
        super().__init__(
            in_features=num_in_features,
            out_features=num_out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        if isinstance(in_features, list):
            self.in_blocks = in_features
            self.out_blocks = out_features if isinstance(out_features, list) else [out_features] * len(in_features)
        else:
            self.in_blocks, self.out_blocks = in_features, out_features

        # setup ordering/masking
        # ordering is a 1D tensor of size of input dimensions (not features per dimension), thus
        # if in_features is a list (i.e. multiple features per dimensions), then we only have as many labels as dimensions
        # but if in_features is a single int, then we have as many labels as features
        ordering_size = out_features if isinstance(self.out_blocks, int) else len(self.out_blocks)
        self.auto_connection = auto_connection
        self.reversed_ordering = reversed_ordering
        self.register_buffer("ordering", torch.empty(ordering_size, device=device, dtype=torch.int))
        self.register_buffer(
            "mask",
            torch.ones(
                ordering_size,
                in_features if isinstance(self.in_blocks, int) else len(self.in_blocks),
                device=device,
                dtype=mask_dtype,
            ),
        )
        self.mode: th.Literal["block", "single"] = "block" if isinstance(self.out_blocks, list) else "single"

    def reorder(
        self,
        inputs_ordering: torch.IntTensor,
        ordering: th.Optional[torch.IntTensor] = None,
        allow_detached_neurons: bool = True,
        highest_ordering_label: th.Optional[int] = None,
        generator: th.Optional[torch.Generator] = None,
    ) -> None:
        """
        (re)computes the output ordering and the mask matrix.

        This function computes the layer's ordering based on the given
        ordering (to be enforced) or the ordering of layer's inputs and a
        random number generator. Optionally you can choose to disallow
        detached neurons, so that the layer ordering labels are chosen
        from higher values than the last layer.

        Args:
            inputs_ordering:
                Ordering of inputs to the layer used for computing new
                layer inputs (randomly out with the generator), and the
                connectivity mask.
            ordering: An optional ordering to be enforced
            allow_detached_neurons:
                If true, the minimum label for this layer's outputs would
                start from zero regardless of whether the previous layer's
                ordering. Else, layer's labels will start from the minimum
                label of inputs.
            highest_ordering_label: Maximum label to use.
            generator: Random number generator.

        Returns:
            None
        """
        if ordering is not None:
            # enforcing ordering and if needed repeating the providded ordering
            # across layer's ordering. (especially used for predicting autoregressive
            # mixture of parameters which requires an output with a size with a multiple
            # of the number of input dimensions.)
            self.ordering.data.copy_(torch.repeat_interleave(ordering, self.ordering.shape[0] // ordering.shape[0]))
        else:
            self.ordering.data.copy_(
                torch.randint(
                    low=0 if allow_detached_neurons else inputs_ordering.min(),
                    high=highest_ordering_label or inputs_ordering.max(),
                    size=self.ordering.shape,
                    generator=generator,
                )
            )
        self.mask.data.copy_(self.connection_operator(inputs_ordering[:, None], self.ordering[None, :]).T)

    @functools.cached_property
    def connection_operator(self):
        if not self.reversed_ordering:
            return torch.less_equal if self.auto_connection else torch.less
        return torch.greater_equal if self.auto_connection else torch.greater

    def reshape_mask(self, mask: torch.Tensor):
        """
        Reshape the mask to the correct shape (if we are in block mode).
        """
        if isinstance(self.in_blocks, int):
            return mask
        else:
            # we have to populate the mask with the correct number of features per dimension
            mask = torch.repeat_interleave(mask, torch.tensor(self.in_blocks).to(mask.device), dim=-1)
            mask = torch.repeat_interleave(mask, torch.tensor(self.out_blocks).to(mask.device), dim=-2)
            return mask

    def permute_mask(self, perm_mat: torch.Tensor) -> torch.Tensor:
        """
        Permutes the mask matrix, with the given permutation matrix.
        Permutations are only allowed in block mode. Otherwise, the result exhibits
        undefined behavior.

        Args:
            permutation: 2D permutation matrix. (num_perms, N x N)

        Returns:
            permuted mask matrix. ([num_perms], N x N])
        """
        return perm_mat @ self.mask.float() @ perm_mat.transpose(-2, -1)

    def forward(
        self,
        inputs: torch.Tensor,
        perm_mat: th.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes masked linear operation.

        Inputs are expected to be batched in the first dimension (2D tensor), or
        a result of a previous masked linear operation (3D tensor, for perm_mat with
        multiple permutations). If the input is a 3D tensor, the dimensions are
        expected to be (num_perms, batch, in_features).

        Args:
            inputs: Input tensor (batched in the first dimensions.)
            perm_mat: Permutation matrix for the output neurons. ([num_perms], N x N)

        Returns:
            A `torch.Tensor` which equals to masked linear operation on inputs, or:
                `inputs @ (self.mask * self.weights).T + self.bias`
            output shape is (batch, out_features) if perm_mat is not None,
        """

        perm_mat = perm_mat.to(self.weight.device) if perm_mat is not None else None
        # inputs = inputs.to(self.weight.device)

        # compute the mask and mask the weights
        mask = self.mask if perm_mat is None else self.permute_mask(perm_mat)

        mask = self.reshape_mask(mask)
        weights = (mask * self.weight).transpose(-2, -1)

        # preprocess shapes
        weights = weights.unsqueeze(0) if weights.ndim == 2 else weights  # add batch dimension if needed
        x = inputs.reshape(-1, inputs.shape[-1])  # flatten inputs (unflatten later)
        weights_batch_size, inputs_batch_size = weights.shape[0], x.shape[0]

        # we have elementwise permutations and we expect the perm_mat to have (and hence the weights)
        # to have a weights_batch_size which divides the inputs_batch_size. We repeat the masked weights
        # for each permutation, to cover all the inputs in batch. e.g. if we have 2 permutations and 4 inputs,
        # we use repeat_interleave to repeat the masked weights 2 times, permute the first 2 inputs with the
        # first permutation, and the second 2 inputs with the second permutation.
        weights = weights.repeat_interleave(inputs_batch_size // weights_batch_size, dim=0)

        results = torch.bmm(x.unsqueeze(1), weights).squeeze(1)
        return results.unflatten(0, inputs.shape[:-1]) + (self.bias if self.bias is not None else 0)
