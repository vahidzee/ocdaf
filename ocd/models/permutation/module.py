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
        device: th.Optional[torch.device] = None,
        num_samples: int = 1,
        soft: bool = True,
        return_noise: bool = False,
        return_matrix: bool = True,
        **kwargs,
    ) -> th.Union[th.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Args:
            device: the device to use (if None, the device of the gamma parameter is used)
            num_samples: the number of samples to draw (default: 1), 0 means no sampling and
                using the current gamma parameter as is (without adding gumbel noise)
            soft: whether to use the soft permutation (default: True)
            return_noise: whether to return the gumbel noise (default: False)
            return_matrix: whether to return the resulting permutations as NxN matrices or as
                a list of ordered indices (default: True)
            *args: arguments to dynamic methods (might be empty, depends on the caller)
            **kwargs: keyword arguments to dynamic methods (might be empty, depends on the caller)

        Returns:
            The resulting permutation matrix or list of ordered indices.
        """
        device = device if device is not None else self.gamma.device
        gumbel_noise = None
        if num_samples:
            gumbel_noise = sample_gumbel_noise(num_samples, self.num_features, self.num_features, device=device)
            gumbel_noise = gumbel_noise * self.gumbel_noise_std
        if soft:
            perm_mat = self.soft_permutation(gumbel_noise=gumbel_noise, *args, **kwargs)
            results = perm_mat if return_matrix else perm_mat.argmax(-1)
        else:
            results = self.hard_permutation(return_matrix=return_matrix, gumbel_noise=gumbel_noise)

        return (results, gumbel_noise) if return_noise else results

    @property  # todo: does not work with the current version of dycode
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
        return 50

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
        return 0.1

    @property  # todo: does not work with the current version of dycode
    @dy.method
    def gumbel_noise_std(self):
        """
        A dynamic method that returns the standard deviation of the Gumbel noise.

        Returns:
            The standard deviation of the Gumbel noise.
        """
        return 2

    def soft_permutation(
        self,
        gamma: th.Optional[torch.Tensor] = None,
        gumbel_noise: th.Optional[torch.Tensor] = None,
        *args,  # for sinkhorn num_iters and temp dynamic methods
        temp: th.Optional[float] = None,
        num_iters: th.Optional[int] = None,
        **kwargs,  # for sinkhorn num_iters and temp dynamic methods
    ) -> torch.Tensor:
        """
        Args:
            gamma: the gamma parameter (if None, the parameterized gamma is used)
            gumbel_noise: the gumbel noise (if None, no noise is added)
            temp: the temperature (if None, the dynamic method sinkhorn_temp is used)
            num_iters: the number of iterations (if None, the dynamic method sinkhorn_num_iters is used)
            *args: arguments to dynamic methods (might be empty, depends on the caller)
            **kwargs: keyword arguments to dynamic methods (might be empty, depends on the caller)

        Returns:
            The resulting permutation matrix.
        """
        gamma = gamma if gamma is not None else self.parameterized_gamma
        temp = temp if temp is not None else self.sinkhorn_temp(*args, **kwargs)
        num_iters = num_iters if num_iters is not None else self.sinkhorn_num_iters(*args, **kwargs)
        # transform gamma with log-sigmoid and temperature
        gamma = torch.nn.functional.logsigmoid(gamma)
        noise = gumbel_noise if gumbel_noise is not None else 0.0
        return sinkhorn((gamma + noise) / temp, num_iters=num_iters)

    def hard_permutation(
        self,
        gamma: th.Optional[torch.Tensor] = None,
        gumbel_noise: th.Optional[torch.Tensor] = None,
        return_matrix: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            gamma: the gamma parameter (if None, the parameterized gamma is used)
            gumbel_noise: the gumbel noise (if None, no noise is added)
            return_matrix: whether to return the resulting permutations as NxN matrices or as
                a list of ordered indices (default: True)

        Returns:
            The resulting permutation matrix or list of ordered indices.
        """
        gamma = gamma if gamma is not None else self.parameterized_gamma
        gamma = gamma + (gumbel_noise if gumbel_noise is not None else 0.0)
        listperm = hungarian(gamma)
        return listperm2matperm(listperm) if return_matrix else listperm

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}"
