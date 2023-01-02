import torch
import typing as th
from .block import MaskedBlock
import functools


class MaskedMLP(torch.nn.ModuleList):
    def __init__(
        self,
        in_features: th.Union[th.List[int], int],
        out_features: th.Union[th.List[int], int],
        layers: th.List[th.Union[th.List[int], int]] = None,
        # residual
        residual: bool = False,
        # blocks
        bias: bool = True,
        activation: th.Optional[str] = "torch.nn.LeakyReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = False,
        batch_norm_args: th.Optional[dict] = None,
        # ordering
        seed: int = 0,
        num_masks: int = 1,
        masks_kind: th.Literal["repeat", "random"] = "repeat",
        # general parameters
        auto_connection: bool = True,
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ):
        """
        Initialize a Masked (autoregressive) Multi-Layer Perceptron.
        """
        super().__init__()
        # process arguments (put them in the right format)
        layers = layers or []
        layers = layers + [out_features]
        if masks_kind == "repeat":
            # if masks_kind is repeat, then in_features and out_features represent blocks of features in the input dimensions
            in_features = in_features if isinstance(in_features, list) else [1] * in_features
            layers = [
                layer if isinstance(layer, list) else [layer // len(in_features)] * len(in_features)
                for layer in layers
            ]

        # set architectural hyperparameters
        self.in_features, self.out_features = in_features, layers[-1]
        # set ordering hyperparameters
        self.masks_kind, self.num_masks, self.__mask_indicator = masks_kind, num_masks, 0
        # set random ordering hyperparameters and setup initial prng
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)

        # initialize layers
        for i, layer_features in enumerate(layers):
            self.append(
                MaskedBlock(
                    in_features=in_features if not i else self[-1].out_blocks,
                    out_features=layer_features,
                    bias=bias,
                    activation=activation if i < len(layers) - 1 else None,
                    auto_connection=True if i < len(layers) - 1 else auto_connection,
                    activation_args=activation_args if i < len(layers) - 1 else None,
                    batch_norm=batch_norm if i < len(layers) - 1 else False,
                    batch_norm_args=batch_norm_args if i < len(layers) - 1 else None,
                    residual=residual if i < len(layers) - 1 else False,
                    device=device,
                    dtype=dtype,
                )
            )

        # initialize ordering
        self.register_buffer(
            "ordering",
            torch.arange(
                self.in_features if isinstance(self.in_features, int) else len(self.in_features),
                dtype=torch.int,
                device=device,
            ),
        )
        self.__base_ordering = self.ordering.clone()
        self.reorder(initialization=True)  # initialize masks and orderings

    def compute_dependencies(
        self,
        inputs: th.Optional[torch.Tensor] = None,
        perm_mat: th.Optional[torch.Tensor] = None,
        mask_index: th.Optional[int] = None,
        vectorize: bool = False,
    ) -> torch.Tensor:
        """
        Compute the data flow dependencies of the model by backpropogating through the model.
        If an output depends on a certain input neuron, then the gradient of the output with respect to the input neuron
        will be non-zero.

        Args:
            mask_index: the index of the mask to use. (None for current mask)
            inputs: the inputs to use for the computation. (default: ones)
            perm_mat: the permutation matrix to use for the computation. (default: None)
            force_reshape_inputs: if True, reshape inputs to the shape (batch_size, -1).
            vectorize: if True, jacobian is computed using vectorized implementation.

        Returns:
            a tensor of shape (out_features, in_features) containing the jacobian of outputs with respect to inputs.
        """
        inputs = (
            torch.ones(
                1,
                self.in_features if isinstance(self.in_features, int) else sum(self.in_features),
                device=self[-1].weight.device,
            )
            if inputs is None
            else inputs.to(self[-1].weight.device)
        )
        inputs.requires_grad = True

        # force evaluation mode
        model_training = self.training  # save training mode to restore later
        self.eval()

        if mask_index is not None:
            # reorder to specified mask_index
            current_mask = self.__mask_indicator  # save current mask to restore later
            self.reorder(
                mask_index=mask_index
            )  # we don't change the mask in forward pass, so backward pass is not affected
        func = functools.partial(self.__call__, perm_mat=perm_mat)

        # compute jacobian of outputs with respect to inputs
        results = torch.autograd.functional.jacobian(func, inputs, vectorize=vectorize)

        if model_training:
            # resotre training mode
            self.train()
        if mask_index is not None:
            # restore previously active mask
            self.reorder(current_mask)

        return results.squeeze()  # remove batch dimensions

    def reorder(
        self,
        ordering: th.Optional[torch.IntTensor] = None,
        seed: th.Optional[int] = None,
        mask_index: th.Optional[int] = None,
        initialization: bool = False,
    ) -> None:
        """
        (re)compute the ordering of inputs and how the dependencies are masked. The underlying ordering
        is set using non-trainable buffers which mask the inputs to each neuron in every layer.

        For instance in an auto-regressive model, we expect the output neuron associated with input dimension
        i to depend only on input dimensions 0, ..., i-1. This is achieved by masking the inputs to each neuron
        in every layer.

        We can assume different orderings of the inputs to the whole model, meaning for instance in a picture,
        we might assume that the pixels are ordered from left to right, or from top to bottom. But to exploit
        the unused weights in the model, we can also assume different input orderings at the same time, and
        when computing the output feature of the model, average the outputs of the different orderings. See
        https://arxiv.org/abs/2111.06377 for more details.

        There are two ways to mask the inputs to each neuron in every layer:
            1. "random" - randomly select what each neuron in the intermediate layers depends on, and compute the
                masks accordingly. This is done by randomly permuting the ordering of the inputs and then masking
                the inputs to each neuron in every layer according to the new ordering.
            2. "repeat" - repeat the same ordering of inputs for each neuron in every layer. If the numb

        This function is called automatically when the model is initialized, and when the forward function is called.

        Args:
            ordering: optional ordering to enforce. If None, then the ordering is set according to num_masks '
                and mask settings.
            seed: optional seed to use for random ordering. (overrides self.seed)
            mask_index: The index of the mask to use. This argument represents different
                ordering assumptions imposed on the inputs to the model. For instance, if the model has 3 masks,
                then mask_index=0 means natural (canonical) ordering, mask_index=1 means first random ordering, and
                mask_index=2 means second random ordering. If mask_index is None, then the next mask is used.

            initialization: If True, the mask is being initialized for the first time. This is used to initialize
                the layers' orderings, without changing the current ordering (which is set by `__mask_indicator`)

        Returns:
            None
        """
        # if there is only one mask, do nothing except if initialization is True (to initialize layers' orderings)
        if self.num_masks == 1 and not initialization and ordering is None:
            return  # nothing to do (performance enhancement for when there is only one mask)

        if mask_index is not None:
            # make sure mask_index is in range of num_masks
            mask_index = mask_index % self.num_masks if self.num_masks else mask_index

        if seed is not None:
            self.seed = seed
            self.generator = torch.Generator().manual_seed(seed)
            self.__mask_indicator = 0  # reset mask indicator to start with the new seed

        # if mask_index is None, use next mask else use specified mask (and remember the original ordering)
        if self.__mask_indicator or mask_index is not None:
            # each ordering corresponds to a random number generotor and a random seed
            self.generator = torch.Generator().manual_seed(
                self.seed + (mask_index if mask_index is not None else self.__mask_indicator)
            )
            if mask_index is not None:  # if mask_index is not None, we are initializing the mask[mask_index]
                self.__mask_indicator = mask_index

        # if ordering is not None, use it and the seed
        if ordering is not None:
            self.__base_ordering = ordering
            self.ordering.data.copy_(ordering)
            self.__mask_indicator = 0  # reset mask indicator to start with the new ordering

        # initialize inputs ordering (mask_index=0 is cannonical ordering the rest are random permutations)
        if not self.num_masks or self.__mask_indicator and ordering is None:
            # generate a random permutation of inputs to use as the ordering for the first layer
            self.ordering.data.copy_(
                torch.randperm(
                    self.in_features if isinstance(self.in_features, int) else len(self.in_features),
                    generator=self.generator,
                )
            )
        elif not self.__mask_indicator:
            # use the canonical ordering (natural ordering) for the first layer (i.e. 0, 1, 2, ..., in_features)
            if ordering is None:
                self.ordering.data.copy_(self.__base_ordering)
            self.generator = torch.Generator().manual_seed(self.seed)  # reset generator to seed

        for i, layer in enumerate(self):
            # if mask_kind is "repeat", the force the ordering of every layer to be the same as the first layer
            force_ordering = self.ordering if self.masks_kind == "repeat" else None
            layer.reorder(
                inputs_ordering=self.ordering if not i else self[i - 1].ordering,
                generator=self.generator,
                # last layer should have the same ordering as the first layer
                ordering=force_ordering if i != len(self) - 1 else self.ordering,
                allow_detached_neurons=True,
                highest_ordering_label=self.in_features
                if isinstance(self.in_features, int)
                else len(self.in_features),
            )

        # if mask_index is None, move to next mask for the next model.reorder() call
        if mask_index is None and not initialization:
            self.__mask_indicator = ((self.__mask_indicator + 1) % self.num_masks) if self.num_masks else 0

    def forward(
        self,
        inputs,
        perm_mat: th.Optional[torch.Tensor] = None,
        mask_index: th.Optional[int] = None,
    ):
        """
        Forward pass of the model. This function is called automatically when the model is called.

        Args:
            inputs: The inputs to the model. The inputs should be a tensor of shape (batch_size, in_features) or
                a tensor of shape (batch_size, *) and force_reshape_inputs=True.
            perm_mat: The permutation matrix to use for the inputs. If None, the inputs are not permuted.
                One can use this argument to learn the permutation matrix for instance using the sinkhorn operator.
            mask_index: The index of the mask to use. (see `reorder` function for more details)
                by default the current ordering of the model is used. (reorder should be called manually to move
                between orderings)

        Returns:
            The output of the model.
        """

        if mask_index is not None:
            current_mask = self.__mask_indicator  # remember current mask to restore it after forward pass
            self.reorder(mask_index=mask_index)
        results = inputs.reshape(-1, inputs.shape[-1])  # flatten all dimensions except the last one
        for layer in self:
            print(results.shape)
            results = layer(results, perm_mat=perm_mat)
        if mask_index is not None:
            # restore the original mask
            self.reorder(mask_index=current_mask)

        # if perm_mat is [NumPerms, N, N], results is now [NumPerms, B*, out_features]
        # but if perm_mat is [N, N], results is [B, out_features], let's make it consistent
        # so that results is always [B*, *, out_features]
        if perm_mat is not None and perm_mat.ndim == 3:
            results = results.transpose(0, 1)
        return results.unflatten(0, inputs.shape[:-1])

    @property
    def orderings(self):
        """
        Returns:
            The orderings of the inputs to each neuron in every layer.
        """
        return [self.ordering] + [layer.ordering for layer in self]

    def extra_repr(self):
        return "num_masks={}, seed={}".format(
            self.num_masks or "inf",
            self.seed,
        )
