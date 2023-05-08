import torch
import typing as th


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


def quantize_soft_permutation(
    soft_permutations: torch.Tensor, hard_permutations: torch.Tensor, return_matrix: bool = True
) -> torch.Tensor:
    assert soft_permutations.shape == hard_permutations.shape
    results = dict()
    perm_mat = HybridJoin.apply(soft_permutations, hard_permutations)
    results["perm_mat"] = perm_mat if return_matrix else perm_mat.argmax(-2)
    results["soft_perm_mats"] = soft_permutations
    return results


def dot_similarity(
    soft_permutations: torch.Tensor,
    hard_permutations: torch.Tensor,
    return_matrix: bool = True,
) -> torch.Tensor:
    results = dict()
    print(soft_permutations.shape, hard_permutations.shape)
    assert soft_permutations.shape == hard_permutations.shape
    dot_prods = torch.sum(soft_permutations * hard_permutations, dim=-1)
    dot_prods = torch.sum(dot_prods, dim=-1)
    results["perm_mat"] = hard_permutations if return_matrix else hard_permutations.argmax(-2)
    results["scores"] = dot_prods
    results["soft_perm_mats"] = soft_permutations
    return results


def sparse_map_approx(
    soft_permutations: torch.Tensor,
    hard_permutations: torch.Tensor,
    return_matrix: bool = True,
    apply_unique: bool = True,
) -> torch.Tensor:
    results = dict()
    # make all the hard_perm_mats unique
    hard_perm_mats = torch.unique(hard_permutations, dim=0) if apply_unique else hard_permutations
    vectorized_soft_mats = soft_permutations.reshape(soft_permutations.shape[0], -1)
    vectorized_hard_mats = hard_perm_mats.reshape(hard_perm_mats.shape[0], -1)
    score_grid = vectorized_soft_mats @ vectorized_hard_mats.T
    # normalize the rows of the score grid
    score_grid = score_grid / torch.sum(score_grid, dim=-1, keepdim=True)
    results["soft_perm_mat"] = soft_permutations
    results["hard_perm_mat"] = hard_perm_mats if return_matrix else hard_perm_mats.argmax(-2)
    results["score_grid"] = score_grid
    return results
