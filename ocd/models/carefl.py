import normflows as nf
import torch
from .masked import MaskedAffineFlow
import dycode as dy
import typing as th
import functools


class CAREFL(torch.nn.Module):
    def __init__(
        self,
        # essential flow args
        base_distribution: th.Union[nf.distributions.BaseDistribution, str],
        base_distribution_args: dict,
        # architecture
        in_features: th.Union[th.List[int], int],
        layers: th.List[th.Union[th.List[int], int]] = None,
        elementwise_perm: bool = False,
        residual: bool = False,
        bias: bool = True,
        activation: th.Optional[str] = "torch.nn.LeakyReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = False,
        batch_norm_args: th.Optional[dict] = None,
        # additional flow args
        additive: bool = False,
        num_transforms: int = 1,
        # ordering
        ordering: th.Optional[torch.IntTensor] = None,
        reversed_ordering: bool = True,
        # general args
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.base_distribution = dy.get_value(base_distribution)(**(base_distribution_args or dict()))
        self.elementwise_perm = elementwise_perm
        # instantiate flows
        self.flows = torch.nn.ModuleList(
            [
                MaskedAffineFlow(
                    in_features=in_features,
                    layers=layers,
                    elementwise_perm=elementwise_perm,
                    residual=residual,
                    bias=bias,
                    activation=activation,
                    activation_args=activation_args,
                    batch_norm=batch_norm,
                    batch_norm_args=batch_norm_args,
                    additive=additive,
                    reversed_ordering=reversed_ordering,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_transforms)
            ]
        )
        if ordering is not None:
            self.reorder(torch.IntTensor(ordering))

    def forward(
        self,
        inputs,
        perm_mat=None,
        elementwise_perm: th.Optional[bool] = None,
        return_intermediate_results: bool = False,
        **kwargs
    ):
        """
        Args:
            inputs: samples from the data distribution (x's)
            perm_mat: permutation matrix to use for the flow (if None, then no permutation is used) (N, D, D)
            elementwise_perm: whether to use elementwise permutation (if None, then self.elementwise_perm is used)
            return_intermediate_results: whether to return the intermediate results (z's) of the flow transformations
        Returns:
            z: transformed samples from the base distribution (z's)
        """
        log_dets, z = 0, inputs.reshape(-1, inputs.shape[-1])
        results = []
        for i, flow in enumerate(self.flows):
            z = z.reshape(-1, inputs.shape[-1])
            if return_intermediate_results:
                results.append(z)
            z, log_det = flow(
                inputs=z, perm_mat=perm_mat, elementwise_perm=elementwise_perm if not i else True, **kwargs
            )
            log_dets += log_det.reshape(-1)
        if perm_mat is not None and not elementwise_perm:
            log_dets = log_dets.reshape(-1, perm_mat.shape[0] if perm_mat.ndim == 3 else 1)
            z = z.reshape(-1, perm_mat.shape[0] if perm_mat.ndim == 3 else 1, inputs.shape[-1])
        z, log_dets = z.unflatten(0, inputs.shape[:-1]), log_dets.unflatten(0, inputs.shape[:-1])
        if return_intermediate_results:
            results.append(z)
            return results
        return z, log_dets

    def compute_dependencies(
        self,
        inputs: torch.Tensor,
        perm_mat: th.Optional[torch.Tensor] = None,
        elementwise_perm: th.Optional[bool] = None,
        forward: bool = True,
        vectorize: bool = True,
        **kwargs
    ) -> torch.Tensor:
        elementwise_perm: bool = elementwise_perm if elementwise_perm is not None else self.elementwise_perm
        # force evaluation mode
        model_training = self.training  # save training mode to restore later
        self.eval()
        func = functools.partial(
            getattr(self, "__call__" if forward else "inverse"),
            perm_mat=perm_mat,
            elementwise_perm=elementwise_perm,
            **kwargs
        )
        # make perm_mat 3d if it is not already
        perm_mat = perm_mat if perm_mat is None or perm_mat.ndim == 3 else perm_mat.unsqueeze(0)

        # compute the jacobian
        jacobian = torch.autograd.functional.jacobian(func, inputs, vectorize=vectorize)
        jacobian = jacobian[0]
        # jacobian for everything else except for matching batch idxs and perm idxs
        # is zero, so we can just sum over the batch dimension and perm dimension to get
        # the jacobian for the
        perm_mat = perm_mat if perm_mat is None or perm_mat.ndim == 3 else perm_mat.unsqueeze(0)
        # its important to note that jacobian is [Output, Input] so to study the effect of permutation
        # we need to figure out what outputs we are interested in
        if not elementwise_perm and perm_mat is not None:
            # depending on the inputs, jacobian is either of shape [batch_size, num_perms, num_dims, batch_size, num_dims]
            # or [batch_size, num_perms, num_dims, batch_size, num_perms, num_dims] either way, we
            # take the mean over the batch dimension so that we either get [num_perms, num_dims, num_dims]
            # or [num_perms, num_dims, num_perms, num_dims] (to study the effect of the permutations)

            jacobian = jacobian.mean(0).sum(2)

            if jacobian.ndim == 4:
                jacobian = jacobian.transpose(1, 2).reshape(-1, jacobian.shape[-1], jacobian.shape[-1])
                jacobian = jacobian[torch.arange(perm_mat.shape[0]).square()]

        elif perm_mat is not None:
            # jacobian is of shape [batch_size, num_dims, batch_size, num_dims], we first split it into
            # [k, num_perms, num_dims, num_dims] and then take the mean over k to get [num_perms, num_dims, num_dims]
            jacobian = jacobian.reshape(-1, perm_mat.shape[0], jacobian.shape[-1], jacobian.shape[-1]).mean(0)

        else:
            # if no perm_mat is given, jacobian is of shape [batch_size, num_dims, batch_size, num_dims]
            # we take the mean over the batch dimension to get [num_dims, num_dims]
            jacobian = jacobian.mean(0).sum(1)

        # restore training mode
        if model_training:
            # resotre training mode
            self.train()

        return jacobian

    def inverse(self, inputs: torch.Tensor, **kwargs) -> th.Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the inverse of the flow.

        Args:
            inputs: Batch of inputs to invert (results of a forward pass)
            **kwargs: Additional arguments (e.g. perm_mat used in forward pass or elementwise_perm)

        Returns:
            Inverted inputs and log determinant of the inverse
        """
        z, log_dets = inputs, 0  # initialize z and log_dets
        # iterate over flows in reverse order and apply inverse
        for i, flow in enumerate(reversed(self.flows)):
            z, log_det = flow.inverse(inputs=z, **kwargs)
            log_dets += log_det  # negative sign is handled in flow.inverse

        return z, log_det

    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        """Sample from the flow

        Args:
          num_samples: Number of samples to generate
        Returns:
          Samples
        """
        z = self.base_distribution(num_samples)[0]
        return self.inverse(z, **kwargs)[0]

    def log_prob(self, x, z=None, log_det=None, **kwargs) -> torch.Tensor:
        """Get log probability for batch

        Args:
          x: Batch of inputs
          z: Batch of latent variables (optional, otherwise computed)
        Returns:
          log probability
        """
        z, log_det = self.forward(x, **kwargs) if z is None else (z, log_det)
        flat_z = z.reshape(-1, z.shape[-1])
        log_base_prob = self.base_distribution.log_prob(flat_z)
        log_base_prob = log_base_prob.reshape(z.shape[:-1])

        return log_base_prob + log_det

    def reorder(self, ordering: th.Optional[torch.IntTensor] = None, **kwargs) -> None:
        if ordering is not None:
            ordering = torch.IntTensor(ordering)
        for flow in self.flows:
            flow.reorder(ordering, **kwargs)

    @property
    def ordering(self) -> torch.IntTensor:
        return self.flows[0].ordering

    @property
    def orderings(self) -> torch.IntTensor:
        return [flow.orderings for flow in self.flows]
