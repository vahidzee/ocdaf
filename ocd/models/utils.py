import torch
import typing as th


def calculate_cumsum(x: th.List[int]) -> th.List[int]:
    """Takes a list of size n and returns a list of size n+1 where 
    the first element is 0 and the rest are the cumsum of the input list"""
    return torch.cumsum(torch.tensor([0] + x[:-1]), dim=0)


def log_prob(
    probas: torch.Tensor,
    cov_features: th.List[int],
    categories: th.Union[torch.Tensor, th.List[th.Any]],
    reduce: bool = False,
    log_probas: bool = True,
) -> torch.Tensor:
    """
    Given the input and the permutation matrix, calculates the log probability of the categories.

    Args:
        probas: (batch_size, sum of [cov_features])
        cov_featuers: a list of the number of categories for each covariate
        categories: a list of observed categories for each covariate that we want to calculate the probability for
        reduce: whether to reduce the log probabilities to a single value
    Returns:
        log_prob: (batch_size, 1) if reduce is True, (batch_size, len(categories)) otherwise
    """
    cummulative = calculate_cumsum(cov_features)
    if isinstance(categories, list):
        categories = torch.tensor(categories, device=probas.device)

    # check if all the elements in categories are less than the corresponding element in cov_features
    assert (
        categories < torch.tensor(cov_features, device=probas.device)
    ).all(), "All category values must be less than the number of categories for that covariate"
    idx = categories + cummulative.to(categories.device)
    ps = torch.vstack([probas[i, idx[i]] for i in range(probas.shape[0])])
    if not log_probas:
        ps = torch.log(ps)
    return ps.sum(-1) if reduce else ps
