import torch
import typing as th
import dypy as dy
from ocd.models.permutation.utils import (
    hungarian,
    sinkhorn,
    sample_gumbel_noise,
    listperm2matperm,
    is_doubly_stochastic,
)


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
        num_samples: int = 1,
        # gamma
        gamma: th.Optional[torch.Tensor] = None,  # to override the current gamma parameter
        # retrieval parameters
        soft: bool = True,
        return_noise: bool = False,
        return_matrix: bool = True,
        # sampling parameters
        gumbel_noise_std: th.Optional[torch.Tensor] = None,
        # sinkhorn parameters
        sinkhorn_num_iters: th.Optional[int] = None,
        sinkhorn_temp: th.Optional[float] = None,
        # general parameters
        device: th.Optional[torch.device] = None,
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
            **kwargs: keyword arguments to dynamic methods (might be empty, depends on the caller)

        Returns:
            The resulting permutation matrix or list of ordered indices.
        """
        device = device if device is not None else self.gamma.device
        gamma = (gamma if gamma is not None else self.parameterized_gamma()).to(device)
        gumbel_noise = None
        if num_samples:
            gumbel_noise = sample_gumbel_noise(num_samples, self.num_features, self.num_features, device=device)
            gumbel_noise_std = gumbel_noise_std if gumbel_noise_std is not None else self.gumbel_noise_std(**kwargs)
            gumbel_noise = gumbel_noise * gumbel_noise_std
        if soft:
            perm_mat = self.soft_permutation(
                gamma=gamma,
                gumbel_noise=gumbel_noise,
                sinkhorn_temp=sinkhorn_temp,
                sinkhorn_num_iters=sinkhorn_num_iters,
                **kwargs,
            )
            results = perm_mat if return_matrix else perm_mat.argmax(-1)
        else:
            results = self.hard_permutation(gamma=gamma, return_matrix=return_matrix, gumbel_noise=gumbel_noise)

        return (results, gumbel_noise) if return_noise else results

    # todo: does not work with the current version of dypy (make it a property later)
    @dy.method
    def parameterized_gamma(self):
        return self.gamma

    @dy.method
    def sinkhorn_num_iters(self, training_module=None, **kwargs) -> int:
        """
        A dynamic method that returns the number of iterations for the Sinkhorn algorithm.

        Args:
            training_module: the training module that calls the dynamic method
            **kwargs: keyword arguments to the dynamic method (might be empty, depends on the caller)

        Returns:
            The number of iterations for the Sinkhorn algorithm.
        """
        return 50

    @dy.method
    def sinkhorn_temp(self, training_module=None, **kwargs) -> float:
        """
        A dynamic method that returns the temperature for the Sinkhorn algorithm.

        Args:

            training_module: the training module that calls the dynamic method
            **kwargs: keyword arguments to the dynamic method (might be empty, depends on the caller)

        Returns:
            The temperature for the Sinkhorn algorithm.
        """
        return 0.1

    # todo: does not work with the current version of dypy (make it a property later)
    @dy.method
    def gumbel_noise_std(self, training_module=None, **kwargs):
        """
        A dynamic method that returns the standard deviation of the Gumbel noise.

        Returns:
            The standard deviation of the Gumbel noise.
        """
        if training_module is None:
            return 0.1
        elif training_module.get_phase() == "maximization":
            return 4
        else:
            return 0.5

    def soft_permutation(
        self,
        gamma: th.Optional[torch.Tensor] = None,
        gumbel_noise: th.Optional[torch.Tensor] = None,
        sinkhorn_temp: th.Optional[float] = None,
        sinkhorn_num_iters: th.Optional[int] = None,
        **kwargs,  # for sinkhorn num_iters and temp dynamic methods
    ) -> torch.Tensor:
        """
        Args:
            gamma: the gamma parameter (if None, the parameterized gamma is used)
            gumbel_noise: the gumbel noise (if None, no noise is added)
            sinkhorn_temp: the temperature (if None, the dynamic method sinkhorn_temp is used)
            sinkhorn_num_iters: the number of iterations (if None, the dynamic method sinkhorn_num_iters is used)
            **kwargs: keyword arguments to dynamic methods (might be empty, depends on the caller)

        Returns:
            The resulting permutation matrix.
        """
        gamma = gamma if gamma is not None else self.parameterized_gamma()
        sinkhorn_temp = sinkhorn_temp if sinkhorn_temp is not None else self.sinkhorn_temp(**kwargs)
        sinkhorn_num_iters = (
            sinkhorn_num_iters if sinkhorn_num_iters is not None else self.sinkhorn_num_iters(**kwargs)
        )
        # transform gamma with log-sigmoid and temperature
        gamma = torch.nn.functional.logsigmoid(gamma)
        noise = gumbel_noise if gumbel_noise is not None else 0.0
        return sinkhorn((gamma + noise) / sinkhorn_temp, num_iters=sinkhorn_num_iters)

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
        gamma = gamma if gamma is not None else self.parameterized_gamma()
        gamma = gamma + (gumbel_noise if gumbel_noise is not None else 0.0)
        listperm = hungarian(gamma)
        return listperm2matperm(listperm) if return_matrix else listperm

    def check_double_stochasticity(
        self,
        num_samples: int = 1000,
        threshold: float = 1e-3,
        return_percentage: bool = True,
        **kwargs,
    ) -> th.Union[torch.Tensor, th.Tuple[torch.Tensor, torch.Tensor]]:
        """
        Checks whether the model forward results in doubly stochastic matrices.

        Args:
            num_samples: the number of samples to check
            threshold: the threshold for the check
            return_percentage: whether to return the percentage of doubly stochastic matrices
                or the boolean tensor and the samples
            **kwargs: keyword arguments to the model forward

        Returns:
            The percentage of doubly stochastic matrices or the boolean tensor and the samples.
        """
        samples = self(num_samples=num_samples, soft=True, return_matrix=True, **kwargs)
        sample_is_doubly_stochastic = is_doubly_stochastic(samples, threshold=threshold)
        if return_percentage:
            return sample_is_doubly_stochastic.float().mean()
        return sample_is_doubly_stochastic, samples

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}"