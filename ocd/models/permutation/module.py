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
    evaluate_permutations,
    translate_idx_ordering,
    is_permutation,
    is_doubly_stochastic,
)
from lightning_toolbox import TrainingModule
import functools
from ocd.models.permutation.hybrid import dot_similarity, quantize_soft_permutation, sparse_map_approx

PERMUTATION_TYPE_OPTIONS = th.Literal[
    "soft", "hard", "hybrid-dot-similarity", "hybrid-quantization", "hybrid-sparse-map-simulator"
]
HYBRID_METHODS = {
    "hybrid-dot-similarity": dot_similarity,
    "hybrid-quantization": quantize_soft_permutation,
    "hybrid-sparse-map-simulator": sparse_map_approx,
}


@dyw.dynamize
class LearnablePermutation(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        force_permutation: th.Optional[th.Union[th.List[int], torch.IntTensor]] = None,
        eps_ignore_permutation_matrix: th.Optional[float] = None,
        eps_ignore_doubly_stochastic: th.Optional[float] = None,
        permutation_type: th.Optional[PERMUTATION_TYPE_OPTIONS] = "soft",
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        num_samples: int = -1,  # -1 for batch size, 0 for no sampling
        num_hard_samples: int = -1,  # -1 for batch size, 0 for no sampling
        hard_from_softs: bool = False,  # hard samples are generated from the same soft samples
        buffer_size: int = 0, # 0 for no buffer
        buffer_replay_prob: float = 1.0,  # out of num_hard_samples portion of samples to be drawn from scratch
        buffer_replace_prob: float = 0.0, # probability of using a new sample instead of a sample from the buffer
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
            eps_ignore_permutation_matrix: the threshold for ignoring soft permutation matrices that have values
                that are not in [eps_permutation, 1 - eps_permutation] (default: None)
            eps_ignore_doubly_stochastic: the threshold for ignoring soft permutation matrices that are not doubly
                stochastic (default: None) (i.e. the row and column sums are in [1 - eps, 1 + eps])
            permutation_type: the type of permutation to use (default: "soft")
                "soft": soft permutation matrices
                "hard": hard permutation matrices
            device: the device to use (if None, the device of the gamma parameter is used)
            dtype: the dtype to use (if None, the dtype of the gamma parameter is used)

            buffer_size: the size of the buffer (default: 0) (used for saving hard permutations)
            buffer_replay_prob: the probability of replaying a sample from the buffer (default: 0.0)
            buffer_replace_prob: the probability of replacing a sample in the buffer (default: 0.0)

        """
        super().__init__()
        self.num_features = num_features
        self.device = device
        self.force_permutation = None
        self.eps_ignore_permutation_matrix = eps_ignore_permutation_matrix
        self.eps_ignore_doubly_stochastic = eps_ignore_doubly_stochastic
        self.permutation_type = permutation_type
        self.num_samples = num_samples
        self.num_hard_samples = num_hard_samples
        self.buffer_size = buffer_size
        self.buffer_replay_prob = buffer_replay_prob
        self.buffer_replace_prob = buffer_replace_prob
        self.hard_from_softs = hard_from_softs
        self.buffer_commits = 0

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
        elif permutation_type.startswith("hybrid"):
            if permutation_type not in HYBRID_METHODS:
                raise Exception(f"Unknown hybrid permutation method: {permutation_type}.")
            return functools.partial(self.hybrid_permutation, method=HYBRID_METHODS[permutation_type])
        else:
            raise Exception(f"Unknown permutation type: {permutation_type}.")

    def _soft_permutations_results(self, soft_permutations: torch.Tensor, return_matrix: bool = True, **kwargs):
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
        return dict(perm_mat=soft_permutations if return_matrix else soft_permutations.argmax(dim=-2))

    def _hard_permutations_results(self, soft_permutations: torch.Tensor, return_matrix: bool = True, **kwargs):
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
        return dict(perm_mat=self.hard_permutation(gamma=soft_permutations, return_matrix=return_matrix))

    def hybrid_permutation(
        self, soft_permutations: torch.Tensor, num_hard_samples: int, method: th.Callable, return_matrix: bool = True
    ):
        hard_permutations = self.sample_hard_permutations(
            soft_permutations=soft_permutations[:num_hard_samples] if not self.hard_from_softs else soft_permutations,
            num_samples=num_hard_samples,
        )
        return method(
            soft_permutations=soft_permutations[num_hard_samples:] if not self.hard_from_softs else soft_permutations,
            hard_permutations=hard_permutations,
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
        permutation_type = permutation_type if permutation_type is not None else self.permutation_type
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

        if not self.hard_from_softs and permutation_type.startswith("hybrid"):
            # hybrid methods use both soft and hard samples
            # thus, if hard_from_softs is False, we don't want to create hard samples from the same soft samples
            # and we need to create more soft samples
            num_samples = num_samples + num_hard_samples
            # the first num_hard_samples of the noises will be used for creating hard permutations
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
            soft_mats = self.soft_permutation(
                gamma=gamma,
                gumbel_noise=gumbel_noise,
                sinkhorn_temp=sinkhorn_temp,
                sinkhorn_num_iters=sinkhorn_num_iters,
                training_module=training_module,
                return_matrix=True,
                **kwargs,
            )

        results = self.get_permutation_method(permutation_type)(
            soft_permutations=soft_mats,
            return_matrix=return_matrix,
            num_hard_samples=num_hard_samples,
        )
        return (results, gumbel_noise) if return_noise else results

    # todo: does not work with the current version of dypy (make it a property later)
    @dyw.method
    def parameterized_gamma(self):
        return self.gamma

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
        return 10

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
        eps_ignore_permutation_matrix: th.Optional[float] = None,
        eps_ignore_doubly_stochastic: th.Optional[float] = None,
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
        # transform gamma with log-sigmoid and temperature
        gamma = torch.nn.functional.logsigmoid(gamma)
        noise = gumbel_noise if gumbel_noise is not None else 0.0
        results = sinkhorn((gamma + noise) / sinkhorn_temp, num_iters=sinkhorn_num_iters)
        return self.ignore_outlier_soft_permutations(
            results, eps_ignore_permutation_matrix, eps_ignore_doubly_stochastic
        )

    def ignore_outlier_soft_permutations(
        self,
        permutations: torch.Tensor,
        eps_ignore_doubly_stochastic: th.Optional[float] = None,
        eps_ignore_permutation_matrix: th.Optional[float] = None,
    ) -> torch.Tensor:
        """
        Ignore outlier matrices and replace them with matrices obtained from the Hungarian algorithm.

        An outlier matrix is one that is too far away from the Birkhoff polytope vertices
        this will be caused in two cases:
            1. The matrix is not doubly stochastic
            2. The matrix is not permutation matrix
        Where the first condition is more strict.

        Args:
            permutations: the soft permutations to check
            eps_ignore_doubly_stochastic: the threshold for the doubly stochastic check (if None, the
                self.eps_ignore_doubly_stochastic is used) (default: None)
                if the resulting value is evaluated to be True, then the matrices are evaluated.
            eps_ignore_permutation_matrix: the threshold for the permutation matrix check (if None, the
                self.eps_ignore_permutation_matrix is used) (default: None)
                if the resulting value is evaluated to be True, then the matrices are evaluated.

        Returns:
            The resulting permutation matrices.
        """
        # idx is a boolean tensor that is True for matrices that are OK
        idx = None
        eps_ignore_doubly_stochastic = (
            eps_ignore_doubly_stochastic
            if eps_ignore_doubly_stochastic is not None
            else self.eps_ignore_doubly_stochastic
        )
        eps_ignore_permutation_matrix = (
            eps_ignore_permutation_matrix
            if eps_ignore_permutation_matrix is not None
            else self.eps_ignore_permutation_matrix
        )
        # ensure matrices have row sum or columns sums in [1 - eps, 1 + eps]
        if eps_ignore_doubly_stochastic:
            idx = is_doubly_stochastic(permutations, threshold=eps_ignore_doubly_stochastic)

        # ensure the matrices have elements between [eps_permutation, 1 - eps_permutation]
        if eps_ignore_permutation_matrix:
            cond = is_permutation(permutations, threshold=eps_ignore_permutation_matrix)
            idx = cond if idx is None else idx & cond

        # For idx we set them using a hard permutation
        if idx is not None and torch.any(~idx):
            hard_permutations = self.hard_permutation(gamma=permutations[~idx])
            results = torch.zeros_like(permutations)
            results[~idx] = hard_permutations
            results[idx] = permutations[idx]
            return results
        return permutations

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
        soft_permutations: torch.Tensor,
        num_samples: th.Optional[int] = None,
        apply_unique: bool = True,
        **kwargs,
    ):
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

    def evaluate_permutations(
        self,
        samples: th.Optional[torch.Tensor] = None,
        num_samples: th.Optional[int] = 1000,
        threshold: float = 1e-3,
        reduce: bool = True,
        **kwargs,
    ) -> th.Dict[str, torch.Tensor]:
        """
        Checks whether the model forward results in doubly stochastic matrices.

        Args:
            num_samples: the number of samples to check
            threshold: the threshold for the check
            return_percentage: whether to return the percentage of doubly stochastic matrices
                or the boolean tensor and the samples
            reduce: whether to reduce the results (take the mean)
            **kwargs: keyword arguments to the model forward

        Returns:
            A dictionary with evaluation results. See the documentation of the
            :func:`evaluate_permutations` function for more details.
        """
        samples = (
            samples if samples is not None else self(num_samples=num_samples, soft=True, return_matrix=True, **kwargs)
        )
        return evaluate_permutations(samples, threshold=threshold, reduce=reduce)

    def extra_repr(self) -> str:
        forced = (
            f', force=[{",".join(str(i)for i in self.force_permutation)}]'
            if self.force_permutation is not None
            else ""
        )
        permutation_type = f", permutation_type={self.permutation_type}" if self.permutation_type != "soft" else ""
        return f"num_features={self.num_features}" + forced + permutation_type
