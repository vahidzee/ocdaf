import torch
import typing as th


# TODO: remove NOT IN THE PAPER
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


# TODO: remove NOT IN THE PAPER
def quantize_soft_permutation(
    soft_permutations: torch.Tensor, hard_permutations: torch.Tensor, return_matrix: bool = True, **kwargs
) -> torch.Tensor:
    """
    Experimental method which didn't end up in the paper.
    """
    assert soft_permutations.shape == hard_permutations.shape
    results = dict()
    perm_mat = HybridJoin.apply(soft_permutations, hard_permutations)
    results["perm_mat"] = perm_mat if return_matrix else perm_mat.argmax(-2)
    results["soft_perm_mats"] = soft_permutations
    return results


def straight_through_soft_permutation(
    soft_permutations: torch.Tensor, hard_permutations: torch.Tensor, return_matrix: bool = True, **kwargs
):
    """
    Implementation of the straight through estimator for the soft permutation matrix.
    """
    results = dict()
    # A trick for straight through estimator
    diff = hard_permutations - soft_permutations
    # turn the gradient off for diff
    perm_mat = soft_permutations + diff.detach()
    results["perm_mat"] = perm_mat if return_matrix else perm_mat.argmax(-2)
    results["soft_perm_mats"] = soft_permutations
    return results


def sparse_map_approx(
    soft_permutations: torch.Tensor,
    hard_permutations: torch.Tensor,
    maximum_basis_size: th.Optional[int] = None,
    return_matrix: bool = True,
    apply_unique: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Implementation of the main algorithm used in the paper for Gumbel Top-k approximation of the proxy loss.
    """
    results = dict()
    # make all the hard_perm_mats unique
    hard_perm_mats = torch.unique(hard_permutations, dim=0) if apply_unique else hard_permutations
    vectorized_soft_mats = soft_permutations.reshape(soft_permutations.shape[0], -1)
    vectorized_hard_mats = hard_perm_mats.reshape(hard_perm_mats.shape[0], -1)
    if vectorized_soft_mats.shape[0] == 1 or vectorized_soft_mats.ndim == 2:
        # not noisy case and using gamma directly
        vectorized_gamma = vectorized_soft_mats.reshape(-1)
        scores = torch.sum(vectorized_gamma * vectorized_hard_mats, dim=-1)

        if maximum_basis_size is not None and len(hard_perm_mats) > maximum_basis_size:
            # keep the indices of the top-self.maximum_basis_size elements of the scores
            _, indices = torch.topk(scores, maximum_basis_size)
            hard_perm_mats = hard_perm_mats[indices]
            scores = scores[indices]
    else:
        score_grid = vectorized_soft_mats @ vectorized_hard_mats.T
    # normalize the rows of the score grid
    score_grid = torch.nn.functional.softmax(scores, dim=-1)
    results["soft_perm_mat"] = soft_permutations
    results["hard_perm_mat"] = hard_perm_mats if return_matrix else hard_perm_mats.argmax(-2)
    results["score_grid"] = score_grid
    return results
