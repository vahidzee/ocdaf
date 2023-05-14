import torch
import typing as th
import dypy.wrappers as dyw
from ocd.models.permutation.utils import (
    hungarian,
    sinkhorn,
    sample_gumbel_noise,
    listperm2matperm,
    evaluate_permutations,
    translate_idx_ordering,
)
from lightning_toolbox import TrainingModule
from .utils import is_permutation, is_doubly_stochastic

PERMUTATION_TYPE_OPTIONS = th.Literal[
    "soft", "hard", "hybrid-dot-similarity", "hybrid-quantization", "hybrid-sparse-map-simulator"
]


class HybridJoin(torch.autograd.Function):
    """
    This function simply passes the backward inputs to both of the inputs
    """

    @staticmethod
    def forward(ctx: th.Any, soft_permutations: torch.Tensor, hard_permutations: torch.Tensor) -> th.Any:
        return hard_permutations

    @staticmethod
    def backward(ctx: th.Any, grad_outputs) -> th.Any:
        return grad_outputs, None


@dyw.dynamize
class LearnablePermutation(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        force_permutation: th.Optional[th.Union[th.List[int], torch.IntTensor]] = None,
        eps_sinkhorn_permutation_matrix: th.Optional[float] = None,
        eps_sinkhorn_doubly_stochastic: th.Optional[float] = None,
        permutation_type: th.Optional[PERMUTATION_TYPE_OPTIONS] = "soft",
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        # Set for limiting the basis
        maximum_basis_size: th.Optional[int] = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.device = device
        self.force_permutation = None
        self.eps_sinkhorn_permutation_matrix = eps_sinkhorn_permutation_matrix
        self.eps_sinkhorn_doubly_stochastic = eps_sinkhorn_doubly_stochastic
        self.permutation_type = permutation_type
        self.maximum_basis_size = maximum_basis_size
        if force_permutation is None:
            # initialize gamma for learnable permutation
            self.gamma = torch.nn.Parameter(torch.randn(num_features, num_features, device=device, dtype=dtype))

            def hook_fn(grad):
                # if grad contains at least one nan, return a tensor of zeros
                # this is for some rare cases where the gradient contains nans
                # and we don't want to spoil the whole training thus far
                return torch.zeros_like(grad) if torch.isnan(grad).any() else grad

            self.gamma.register_hook(hook_fn)
        else:
            # used for debugging purposes only and is None by default
            self.force_permutation = translate_idx_ordering(force_permutation)

    def forward(
        self,
        batch_size: int = 1,
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
        training_module: th.Optional[TrainingModule] = None,
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
        num_samples = batch_size
        num_hard_samples = 0
        num_soft_samples = 0
        permutation_type = permutation_type if permutation_type is not None else self.permutation_type

        if permutation_type == "hybrid-sparse-map-simulator":
            num_soft_samples = batch_size
            num_hard_samples = batch_size
            num_samples = num_samples + num_hard_samples

        # Generate Gumbel noise values
        if num_samples:
            gumbel_noise = sample_gumbel_noise(num_samples, self.num_features, self.num_features, device=device)
            gumbel_noise_std = (
                gumbel_noise_std
                if gumbel_noise_std is not None
                else self.gumbel_noise_std(training_module=training_module, **kwargs)
            )
            gumbel_noise = gumbel_noise * gumbel_noise_std

        # Set up an empty dictionary results
        results = {}
        if permutation_type == "soft":
            if num_hard_samples > 0:
                raise ValueError("Cannot use hard samples with soft permutation, set num_hard_samples to 0.")
            perm_mat = self.soft_permutation(
                gamma=gamma,
                gumbel_noise=gumbel_noise,
                sinkhorn_temp=sinkhorn_temp,
                sinkhorn_num_iters=sinkhorn_num_iters,
                training_module=training_module,
                **kwargs,
            )
            # TODO: Check the following out closely! It might cause errors!
            results["perm_mat"] = perm_mat if return_matrix else perm_mat.argmax(-2)
        elif permutation_type == "hard":
            if num_soft_samples > 0:
                raise ValueError("Cannot use hard samples with soft permutation, set num_hard_samples to 0.")

            results["perm_mat"] = self.hard_permutation(
                gamma=gamma, return_matrix=return_matrix, gumbel_noise=gumbel_noise
            )
        elif permutation_type.startswith("hybrid"):
            if permutation_type == "hybrid-quantization":
                soft_perm_mats = self.soft_permutation(
                    gamma=gamma,
                    gumbel_noise=gumbel_noise,
                    sinkhorn_temp=sinkhorn_temp,
                    sinkhorn_num_iters=sinkhorn_num_iters,
                    training_module=training_module,
                    **kwargs,
                )
                hard_perm_mats = self.hard_permutation(gamma=gamma, return_matrix=True, gumbel_noise=gumbel_noise)

                if max(num_hard_samples, num_soft_samples) > 0:
                    raise ValueError(
                        "For hybrid Quantization num_hard_samples and num_soft_samples should be set to 0."
                    )
                
                # A trick for straight through estimator
                diff = hard_perm_mats - soft_perm_mats
                # turn the gradient off for diff
                perm_mat = soft_perm_mats + diff.detach()
                results["perm_mat"] = perm_mat if return_matrix else perm_mat.argmax(-2)
                results["soft_perm_mats"] = soft_perm_mats
            elif permutation_type == "hybrid-dot-similarity":
                soft_perm_mats = self.soft_permutation(
                    gamma=gamma,
                    gumbel_noise=gumbel_noise,
                    sinkhorn_temp=sinkhorn_temp,
                    sinkhorn_num_iters=sinkhorn_num_iters,
                    training_module=training_module,
                    **kwargs,
                )
                hard_perm_mats = self.hard_permutation(gamma=gamma, return_matrix=True, gumbel_noise=gumbel_noise)

                if max(num_hard_samples, num_soft_samples) > 0:
                    raise ValueError(
                        "For hybrid Dot Similarity num_hard_samples and num_soft_samples should be set to 0."
                    )
                dot_prods = torch.sum(soft_perm_mats * hard_perm_mats, dim=-1)
                dot_prods = torch.sum(dot_prods, dim=-1)
                results["perm_mat"] = hard_perm_mats if return_matrix else hard_perm_mats.argmax(-2)
                results["scores"] = dot_prods
                results["soft_perm_mats"] = soft_perm_mats
            elif permutation_type == "hybrid-sparse-map-simulator":
                # Sample a set of hard permutations
                hard_perm_mats = self.hard_permutation(
                    gamma=gamma, return_matrix=True, gumbel_noise=gumbel_noise[num_soft_samples:]
                )
                # make all the hard_perm_mats unique to obtain H_k which is the the approximate Boltzmann support
                hard_perm_mats = torch.unique(hard_perm_mats, dim=0)
                
                # Calculate the energies of the Boltzmann support
                vectorized_hard_mats = hard_perm_mats.reshape(hard_perm_mats.shape[0], -1)
                vectorized_gamma = self.parameterized_gamma().reshape(-1)
                scores = torch.sum(vectorized_gamma * vectorized_hard_mats, dim=-1)
                
                if self.maximum_basis_size is not None and len(hard_perm_mats) > self.maximum_basis_size:
                    # keep the indices of the top-self.maximum_basis_size elements of the scores
                    _, indices = torch.topk(scores, self.maximum_basis_size)
                    hard_perm_mats = hard_perm_mats[indices]
                    scores = scores[indices]
                
                # Approximate the actual pmf using softmax
                scores = torch.nn.functional.softmax(scores, dim=-1)
                
                results["hard_perm_mat"] = hard_perm_mats
                results["scores"] = scores
            else:
                raise Exception(f"Unknown hybrid permutation type: {permutation_type}")
        else:
            raise Exception(f"Unknown permutation type: {permutation_type}.")

        return (results, gumbel_noise) if return_noise else results

    def get_permutation_without_noise(self):
        """
        This function simply returns the permutation matrix without any noise.
        This can be used for visualizing the Gamma parameters of the model for example.
        """
        return self.soft_permutation().detach().cpu().numpy()

    # todo: does not work with the current version of dypy (make it a property later)
    @dyw.method
    def parameterized_gamma(self):
        if self.permutation_type == "hybrid-sparse-map-simulator":
            return torch.nn.functional.sigmoid(self.gamma)
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
        eps_sinkhorn_permutation_matrix: th.Optional[float] = None,
        eps_sinkhorn_doubly_stochastic: th.Optional[float] = None,
        training_module: th.Optional[TrainingModule] = None,
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
            The resulting permutation matrices, and the percentage of ones that are replaced by hard ones.
        """
        gamma = gamma if gamma is not None else self.parameterized_gamma()
        sinkhorn_temp = sinkhorn_temp if sinkhorn_temp is not None else self.sinkhorn_temp(training_module=training_module, **kwargs)
        sinkhorn_num_iters = (
            sinkhorn_num_iters if sinkhorn_num_iters is not None else self.sinkhorn_num_iters(training_module=training_module, **kwargs)
        )
        # transform gamma with log-sigmoid and temperature
        # gamma = torch.nn.functional.logsigmoid(gamma)
        noise = gumbel_noise if gumbel_noise is not None else 0.0
        all_mats = sinkhorn((gamma + noise) / sinkhorn_temp, num_iters=sinkhorn_num_iters)

        # Now we will ignore outlier matrices and replace them with matrices obtained
        # from the Hungarian algorithm.
        # An outlier matrix is one that is too far away from the Birkhoff polytope vertices
        # this will be caused in two cases:
        # 1. The matrix is not doubly stochastic
        # 2. The matrix is not permutation matrix
        # The first condition is more strict

        # idx contains all the "BAD" matrices
        idx = None

        # ignore matrices that have row sum or columns sums not in [1 - eps, 1 + eps]
        if self.eps_sinkhorn_doubly_stochastic is not None:
            eps = eps_sinkhorn_doubly_stochastic or self.eps_sinkhorn_doubly_stochastic
            idx = is_doubly_stochastic(all_mats, threshold=eps)

        # ignore the matrices that have elements between [eps_permutation, 1 - eps_permutation]
        if self.eps_sinkhorn_permutation_matrix is not None:
            eps = eps_sinkhorn_permutation_matrix or self.eps_sinkhorn_permutation_matrix
            cond = is_permutation(all_mats, threshold=eps)
            idx = cond if idx is None else idx & cond

        # For idx we set them using a hard permutation
        if idx is not None and torch.any(~idx):
            if isinstance(noise, torch.Tensor):
                noise = noise[~idx]
            hard_permutations = self.hard_permutation(gamma=gamma, gumbel_noise=noise)
            surrogate_matrices = torch.zeros_like(all_mats)
            surrogate_matrices[~idx] = hard_permutations
            surrogate_matrices[idx] = all_mats[idx]
            ret = surrogate_matrices
        else:
            ret = all_mats

        return ret

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
        if return_matrix:
            return listperm2matperm(listperm, device=gamma.device)
        return translate_idx_ordering(listperm.tolist())

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
        return evaluate_permutations(samples, threshold=threshold)

    def extra_repr(self) -> str:
        forced = (
            f', force=[{",".join(str(i)for i in self.force_permutation)}]'
            if self.force_permutation is not None
            else ""
        )
        return f"num_features={self.num_features}" + forced