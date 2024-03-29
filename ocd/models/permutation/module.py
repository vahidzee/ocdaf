import torch
import typing as th
import dypy.wrappers as dyw
import numpy as np
import random
from ocd.models.permutation.utils import (
    hungarian,
    sinkhorn,
    sample_gumbel_noise,
    listperm2matperm,
    translate_idx_ordering,
)
from lightning_toolbox import TrainingModule
import functools
from ocd.models.permutation.methods import (
    gumbel_topk,
    straight_through,
)

PERMUTATION_TYPE_OPTIONS = th.Literal[
    "soft",
    "hard",
    "gumbel-topk",
    "gumbel-topk-noisy",
    "straight-through",
    "straight-through-noisy",
]
HYBRID_METHODS = {
    "gumbel-topk": gumbel_topk,
    "gumbel-topk-noisy": gumbel_topk,
    "straight-through": straight_through,
    "straight-through-noisy": straight_through,
}


@dyw.dynamize
class LearnablePermutation(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        force_permutation: th.Optional[th.Union[th.List[int], torch.IntTensor]] = None,
        permutation_type: th.Optional[PERMUTATION_TYPE_OPTIONS] = "soft",
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        num_samples: int = -1,  # -1 for batch size, 0 for no sampling
        num_hard_samples: int = -1,  # -1 for batch size, 0 for no sampling
        hard_from_softs: bool = True,  # hard samples are generated from the same soft samples
        buffer_size: int = 0,  # 0 for no buffer
        buffer_replay_prob: float = 1.0,  # out of num_hard_samples portion of samples to be drawn from scratch
        buffer_replace_prob: float = 0.0,  # probability of using a new sample instead of a sample from the buffer
        # TODO: clean up these parameters
        maximum_basis_size: th.Optional[int] = None,
    ):
        """
        Learnable permutation matrices.

        For computational efficiency, since since the hungarian algorithm is O(N^3), you might want to configure
        the buffer size and the buffer probabilities, so that the hungarian algorithm is not called too often.

        Every time a new hard permutation matrix is required, the following happens:
            1. If the buffer is not full, then the new batch of soft_permutations are converted to hard permutations
                and added to the buffer.
            2. If the buffer is full, then with `1-buffer_replace_prob` samples are drawn from the buffer, and with
                `buffer_replay_prob` next step is executed.
            3. A binomial random variable is sampled with probability `1-buffer_replay_prob` and and the number of samples,
                to determine how many new hard permutations should be selected from the soft permutations. The rest of the
                samples are drawn from the buffer, and the new samples are added to the buffer.

        Args:
            num_features: the number of features (i.e. the dimension of the permutation matrices)
            force_permutation: a permutation to force (used for debugging purposes only) and is a list (or a list of lists)
                of indices (default: None)
                For example if `num_features = 3`, then `force_permutation = [2, 0, 1]` means that the third feature
                should be the first, the first should be the second, and the second should be the third.
                Or [[2, 0, 1], [1, 2, 0]] results in a batch of two permutations.
            permutation_type: the type of permutation to use (default: "soft")
                "soft": soft permutation matrices
                "hard": hard permutation matrices
            device: the device to use (if None, the device of the gamma parameter is used)
            dtype: the dtype to use (if None, the dtype of the gamma parameter is used)

            buffer_size: the size of the buffer (default: 0) (used for saving hard permutations)
            buffer_replay_prob: the probability of replaying a sample from the buffer (default: 0.0)
            buffer_replace_prob: the probability of replacing a sample in the buffer (default: 0.0)
            num_samples: the number of samples to draw (default: -1), -1 means using the batch size
            num_hard_samples: the number of hard samples to draw (default: -1), -1 means using the batch size
            hard_from_softs: whether to generate hard samples from the same soft samples (default: False)
            maximum_basis_size: the maximum number of basis to use for the sparse map approximation using
                 topk algorithm (default: None)
        """
        super().__init__()
        self.num_features = num_features
        self.device = device
        self.force_permutation = None
        self.permutation_type = permutation_type
        self.num_samples = num_samples
        self.num_hard_samples = num_hard_samples
        self.buffer_size = buffer_size
        self.buffer_replay_prob = buffer_replay_prob
        self.buffer_replace_prob = buffer_replace_prob
        self.hard_from_softs = hard_from_softs
        self.buffer_commits = 0
        self.maximum_basis_size = maximum_basis_size

        if force_permutation is None:
            # initialize gamma for learnable permutation
            self.gamma = torch.nn.Parameter(torch.randn(num_features, num_features, device=device, dtype=dtype))

            # TODO: remove this hook after debugging
            def hook_fn(grad):
                # if grad contains at least one nan, return a tensor of zeros
                # this is for some rare cases where the gradient contains nans
                # and we don't want to spoil the whole training thus far
                return torch.zeros_like(grad) if torch.isnan(grad).any() else grad

            self.gamma.register_hook(hook_fn)
        else:
            # used for debugging purposes only and is None by default
            self.force_permutation = translate_idx_ordering(force_permutation)

        self.register_buffer(
            "buffer", torch.empty((buffer_size, num_features, num_features), device=device, dtype=dtype)
        )

    @functools.cached_property
    def permutation_method(self):
        """Cached property for the permutation method. For faster access."""
        return self.get_permutation_method(self.permutation_type)

    def get_permutation_method(self, permutation_type: th.Optional[PERMUTATION_TYPE_OPTIONS] = None) -> th.Callable:
        """
        Returns a callable that takes on soft_permutations and returns the permutation matrices,
        according to the given permutation type.

        Args:
            permutation_type: the type of permutation to use, if None, the current permutation type is used.
                For faster access to the current permutation type, use the cached property `permutation_method`.
        Returns:
            A callable that takes on soft_permutations and returns the permutation matrices.
        """
        if permutation_type is None:
            return self.permutation_method
        if permutation_type == "soft":
            return self._soft_permutations_results
        elif permutation_type == "hard":
            return self._hard_permutations_results
        elif permutation_type not in HYBRID_METHODS:
            raise Exception(f"Unknown permutation method: {permutation_type}.")
        return functools.partial(
            self.hybrid_permutation, method=HYBRID_METHODS[permutation_type.replace("-noisy", "")]
        )

    def _soft_permutations_results(
        self,
        gumbel_noise: torch.Tensor,
        gamma: th.Optional[torch.Tensor] = None,
        return_matrix: bool = True,
        sinkhorn_num_iters: th.Optional[int] = None,
        sinkhorn_temp: th.Optional[float] = None,
        **kwargs,
    ):
        """
        Utility function for returning the results of soft permutations.

        Args:
            soft_permutations: the soft permutations
            return_matrix: whether to return the resulting permutations as NxN matrices or as
                a list of ordered indices (default: True)
            **kwargs: additional keywords are ignored.

        Returns:
            A dictionary with the resulting permutations (perm_mat).
        """
        soft_permutations = self.soft_permutation(
            gamma=gamma,
            gumbel_noise=gumbel_noise,
            sinkhorn_temp=sinkhorn_temp,
            sinkhorn_num_iters=sinkhorn_num_iters,
            return_matrix=return_matrix,
        )
        return dict(perm_mat=soft_permutations if return_matrix else soft_permutations.argmax(dim=-2))

    def _hard_permutations_results(
        self,
        gumbel_noise: torch.Tensor,
        gamma: th.Optional[torch.Tensor] = None,
        return_matrix: bool = True,
        **kwargs,
    ):
        """
        Utility function for returning the results of hard permutations.

        Args:
            soft_permutations: the soft permutations
            return_matrix: whether to return the resulting permutations as NxN matrices or as
                a list of ordered indices (default: True)
            **kwargs: additional keywords are ignored.

        Returns:
            A dictionary with the resulting permutations (perm_mat).
        """
        return dict(
            perm_mat=self.hard_permutation(gamma=gamma, gumbel_noise=gumbel_noise, return_matrix=return_matrix)
        )

    def hybrid_permutation(
        self,
        num_hard_samples: int,
        gumbel_noise: torch.Tensor,
        method: th.Callable,
        return_matrix: bool = True,
        # sampling parameters
        gamma: th.Optional[torch.Tensor] = None,  # to override the current gamma parameter
        # sinkhorn parameters
        sinkhorn_num_iters: th.Optional[int] = None,
        sinkhorn_temp: th.Optional[float] = None,
        # general parameters
        **kwargs,
    ):
        """
        Caller function for hybrid methods (that combine hard and soft permutations). This function
        prepares the soft and hard permutations and calls the given method (e.g. gumbel_topk). To
        see the details of the method, see the corresponding function in `ocd/models/permutation/methods.py`.
        """
        if self.permutation_type.endswith("-noisy"):
            soft_permutations = self.soft_permutation(
                gamma=gamma,
                gumbel_noise=(
                    gumbel_noise[: len(gumbel_noise) - num_hard_samples] if not self.hard_from_softs else gumbel_noise
                ),
                sinkhorn_temp=sinkhorn_temp,
                sinkhorn_num_iters=sinkhorn_num_iters,
                return_matrix=True,
            )
        else:
            soft_permutations = self.parameterized_gamma() if gamma is None else gamma

        hard_permutations = self.sample_hard_permutations(
            num_samples=num_hard_samples,
            gumbel_noise=(
                gumbel_noise[len(gumbel_noise) - num_hard_samples :] if not self.hard_from_softs else gumbel_noise
            ),
            gamma=gamma,
        )
        return method(
            soft_permutations=soft_permutations if not self.hard_from_softs else soft_permutations,
            hard_permutations=hard_permutations,
            maximum_basis_size=self.maximum_basis_size,
            return_matrix=return_matrix,
        )

    def forward(
        self,
        batch_size: int,
        # gamma
        gamma: th.Optional[torch.Tensor] = None,  # to override the current gamma parameter
        # force permutation
        force_permutation: th.Optional[th.Union[th.List[int], torch.IntTensor]] = None,
        # retrieval parameters
        permutation_type: th.Optional[PERMUTATION_TYPE_OPTIONS] = None,
        return_noise: bool = False,
        return_matrix: bool = True,
        # sampling parameters
        gumbel_noise_std: th.Optional[torch.Tensor] = None,
        # sinkhorn parameters
        sinkhorn_num_iters: th.Optional[int] = None,
        sinkhorn_temp: th.Optional[float] = None,
        # general parameters
        device: th.Optional[torch.device] = None,
        training_module: th.Optional["TrainingModule"] = None,
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
        perm_type = permutation_type if permutation_type is not None else self.permutation_type
        # force permutation if given (used for debugging purposes only)
        if force_permutation is not None or self.force_permutation is not None:
            force_permutation = force_permutation if force_permutation is not None else self.force_permutation
            results = listperm2matperm(force_permutation, device=device)
            return (results, None) if return_noise else results  # for consistency with the other return statements

        # otherwise, use the current gamma parameter (or the given one) to compute the permutation
        device = device if device is not None else self.gamma.device
        gumbel_noise = None
        gamma = (gamma if gamma is not None else self.parameterized_gamma()).to(device)

        # depending on the method, num_samples might be different
        num_samples = self.num_samples if self.num_samples >= 0 else batch_size
        num_hard_samples = self.num_hard_samples if self.num_hard_samples >= 0 else batch_size

        if not self.hard_from_softs and perm_type in HYBRID_METHODS:
            # hybrid methods use both soft and hard samples
            # thus, if hard_from_softs is False, we don't want to create hard samples from the same soft samples
            # and we need to create more soft samples
            num_samples = num_samples + num_hard_samples
            # the last num_hard_samples of the noises will be used for creating hard permutations
            # the rest of the noises will be used for creating soft permutations

        # Generate gumbel noise values and subsequently the soft permutation matrices
        if num_samples:
            gumbel_noise = sample_gumbel_noise(num_samples, self.num_features, self.num_features, device=device)
            gumbel_noise_std = (
                gumbel_noise_std
                if gumbel_noise_std is not None
                else self.gumbel_noise_std(training_module=training_module, **kwargs)
            )
            gumbel_noise = gumbel_noise * gumbel_noise_std

        results = self.get_permutation_method(permutation_type)(
            gamma=gamma,
            gumbel_noise=gumbel_noise,
            sinkhorn_temp=sinkhorn_temp,
            sinkhorn_num_iters=sinkhorn_num_iters,
            training_module=training_module,
            return_matrix=return_matrix,
            num_hard_samples=num_hard_samples,
            **kwargs,
        )
        return (results, gumbel_noise) if return_noise else results

    # todo: does not work with the current version of dypy (make it a property later)
    @dyw.method
    def parameterized_gamma(self):
        if self.permutation_type == "gumbel-topk":
            return torch.sigmoid(self.gamma)
        else:
            return -torch.nn.functional.logsigmoid(self.gamma)

    @dyw.method
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

    @dyw.method
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
    @dyw.method
    def gumbel_noise_std(self, training_module=None, **kwargs):
        """
        A dynamic method that returns the standard deviation of the Gumbel noise.

        Returns:
            The standard deviation of the Gumbel noise.
        """
        if training_module is None:
            ret = 1
        elif training_module.current_phase == "maximization":
            ret = 5
        elif training_module.current_phase == "expectation":
            ret = 5
        else:
            raise ValueError(f"Unknown phase: {training_module.current_phase}")
        return ret

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
            The resulting permutation matrices.
        """
        gamma = gamma if gamma is not None else self.parameterized_gamma()
        sinkhorn_temp = sinkhorn_temp if sinkhorn_temp is not None else self.sinkhorn_temp(**kwargs)
        sinkhorn_num_iters = (
            sinkhorn_num_iters if sinkhorn_num_iters is not None else self.sinkhorn_num_iters(**kwargs)
        )
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
        # TODO: add buffer and sampling here
        if return_matrix:
            return listperm2matperm(listperm, device=gamma.device)
        return translate_idx_ordering(listperm.tolist())

    def update_buffer(self, permutations: torch.Tensor, apply_unique: bool = True, **kwargs):
        """
        Updates the buffer with the given permutations.

        Args:
            permutations: the permutations to add to the buffer
            apply_unique: whether to enforce uniqueness of the permutations (default: True)
            **kwargs: keyword arguments to the dynamic methods (might be empty, depends on the caller)
        """
        if not self.buffer_size:
            return
        permutations = permutations.detach()  # detach to avoid memory leaks
        # if the buffer is not full, add the permutations to the buffer
        all_perms = torch.cat([self.buffer[: self.buffer_commits], permutations], dim=0)
        if apply_unique:
            all_perms = all_perms.unique(dim=0)

        if self.buffer_commits < self.buffer_size:
            self.buffer[: min(len(all_perms), self.buffer_size)] = all_perms[
                torch.randperm(len(all_perms))[: self.buffer_size]
            ]
            self.buffer_commits = min(len(all_perms), self.buffer_size)
        else:
            self.buffer = all_perms[torch.randperm(len(all_perms))[: self.buffer_size]]

    def sample_hard_permutations(
        self,
        gamma: torch.Tensor,
        gumbel_noise: torch.Tensor,
        num_samples: th.Optional[int] = None,
        apply_unique: bool = True,
        **kwargs,
    ):
        soft_permutations = gamma + gumbel_noise
        num_samples = num_samples if num_samples is not None else self.num_hard_samples
        num_samples = num_samples if num_samples > 0 else soft_permutations.shape[0]
        # sample directly from soft permutations
        assert len(soft_permutations) >= num_samples
        if not self.buffer_size:
            return self.hard_permutation(gamma=soft_permutations, return_matrix=True)[:num_samples]

        assert self.buffer_size >= num_samples
        # fill the buffer if it is empty and sample from it
        if self.buffer_commits < self.buffer_size:
            self.update_buffer(
                self.hard_permutation(gamma=soft_permutations, return_matrix=True), apply_unique=apply_unique
            )

        if self.buffer_commits < self.buffer_size or (
            self.buffer_replace_prob and np.random.rand() >= 1 - self.buffer_replace_prob
        ):
            return self.buffer[torch.randperm(self.buffer_commits)[:num_samples]]

        # sample from the buffer and the soft permutations
        num_new = np.random.binomial(num_samples, 1 - self.buffer_replay_prob)

        # select num_new indices from soft_permutations and convert them to hard permutations
        new_perms = self.hard_permutation(
            gamma=soft_permutations[torch.randperm(len(soft_permutations))[:num_new]], return_matrix=True
        )
        # select num_samples - num_new indices from the buffer
        old_perms = self.buffer[torch.randperm(self.buffer_size)[: num_samples - num_new]]
        # update the buffer
        self.update_buffer(new_perms, apply_unique=apply_unique)
        return torch.cat([new_perms, old_perms], dim=0)

    def extra_repr(self) -> str:
        forced = (
            f', force=[{",".join(str(i)for i in self.force_permutation)}]'
            if self.force_permutation is not None
            else ""
        )
        permutation_type = f", permutation_type={self.permutation_type}" if self.permutation_type != "soft" else ""
        return f"num_features={self.num_features}" + forced + permutation_type
