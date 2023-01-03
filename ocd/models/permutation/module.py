import torch
import typing as th
import dycode as dy
from ocd.models.permutation.utils import hungarian, sinkhorn, sample_gumbel_noise, listperm2matperm


@dy.dynamize
class LearnablePermutation(torch.nn.Module):
    def __init__(
        self, num_features: int, device: th.Optional[torch.device] = None, dtype: th.Optional[torch.dtype] = None
    ):
        super().__init__()
        self.num_features = num_features
        self.device = device
        self.gamma = torch.nn.Parameter(torch.randn(num_features, num_features, device=device, dtype=dtype))

    def forward(
        self,
        *args,
        inputs: th.Optional[torch.Tensor] = None,
        num_samples: int = 1,
        soft: bool = True,
        return_noise: bool = False,
        return_matrix: bool = True,
        **kwargs,
    ):
        device = inputs.device if inputs is not None else self.gamma.device
        if soft:
            gumbel_noise = (
                sample_gumbel_noise(num_samples, self.num_features, self.num_features, device=device)
                if num_samples
                else None
            )
            perm_mat = self.soft_permutation(gumbel_noise=gumbel_noise, *args, **kwargs)
            results = perm_mat if return_matrix else perm_mat.argmax(-1)
        else:
            results = self.hard_permutation(return_matrix=return_matrix)

        return (results, gumbel_noise) if return_noise else results

    @property
    @dy.method
    def parameterized_gamma(self):
        return self.gamma

    @dy.method
    def sinkhorn_num_iters(self, *args, training_module=None, **kwargs) -> int:
        """
        A dynamic method that returns the number of iterations for the Sinkhorn algorithm.

        Args:
            *args: arguments to the dynamic method (might be empty, depends on the caller)
            training_module: the training module that calls the dynamic method
            **kwargs: keyword arguments to the dynamic method (might be empty, depends on the caller)

        Returns:
            The number of iterations for the Sinkhorn algorithm.
        """
        return 10

    @dy.method
    def sinkhorn_temp(self, *args, training_module=None, **kwargs) -> float:
        """
        A dynamic method that returns the temperature for the Sinkhorn algorithm.

        Args:
            *args: arguments to the dynamic method (might be empty, depends on the caller)
            training_module: the training module that calls the dynamic method
            **kwargs: keyword arguments to the dynamic method (might be empty, depends on the caller)

        Returns:
            The temperature for the Sinkhorn algorithm.
        """
        return 1.0

    def soft_permutation(
        self,
        gamma: th.Optional[torch.Tensor] = None,
        gumbel_noise: th.Optional[torch.Tensor] = None,
        *args,  # for sinkhorn num_iters and temp dynamic methods
        temp: th.Optional[float] = None,
        num_iters: th.Optional[int] = None,
        **kwargs,  # for sinkhorn num_iters and temp dynamic methods
    ) -> torch.Tensor:
        gamma = gamma if gamma is not None else self.parameterized_gamma
        temp = temp if temp is not None else self.sinkhorn_temp(*args, **kwargs)
        num_iters = num_iters if num_iters is not None else self.sinkhorn_num_iters(*args, **kwargs)
        # transform gamma with log-sigmoid and temperature
        gamma = torch.nn.functional.logsigmoid(gamma + (gumbel_noise if gumbel_noise is not None else 0.0) / temp)
        return sinkhorn(gamma, num_iters=num_iters)

    def hard_permutation(self, gamma: th.Optional[torch.Tensor] = None, return_matrix: bool = False) -> torch.Tensor:
        gamma = gamma if gamma is not None else self.parameterized_gamma
        listperm = hungarian(gamma)
        return listperm2matperm(listperm) if return_matrix else listperm

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}"
