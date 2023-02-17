import torch
import typing as th

# most of the codes are humbly borrowed/adapted from
# https://github.com/sharpenb/Differentiable-DAG-Sampling/tree/44f96769a729efc99bdd16c9b00deee4077a76b2


def sample_gumbel_noise(*args, eps=1e-20, **kwargs):
    """Samples arbitrary-shaped standard gumbel variables.
    Args:
        shape: list of integers
        eps: float, for numerical stability
    Returns:
        A sample of standard Gumbel random variables
    """

    u = torch.rand(*args, **kwargs).float()
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_log_prob(gumbel_noise: torch.Tensor) -> torch.Tensor:
    """Computes the log-probability of a sample of Gumbel random variables.
    Args:
        gumbel_noise: a sample of Gumbel random variables
    Returns:
        The log-probability of the sample
    """
    return (-gumbel_noise - torch.exp(-gumbel_noise)).sum(dim=[-1, -2])


def sinkhorn(log_alpha, num_iters=20):
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the succesive row and column
    normalization.
        -To ensure positivity, the effective input to sinkhorn has to be
        exp(log_alpha) (elementwise).
        -However, for stability, sinkhorn works in the log-space. It is only at
        return time that entries are exponentiated.
        
    [1] https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-21/issue-2/\
        Concerning-nonnegative\-matrices-and-doubly-stochastic-matrices

    Args:
        log_alpha: 2D tensor (a matrix of shape [N, N])
            or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        num_iters: number of sinkhorn iterations (in practice, as little as 20
            iterations are needed to achieve decent convergence for N~100)
    Returns:
        A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
            converted to 3D tensors with batch_size equals to 1)
    """
    # print(log_alpha)
    n = log_alpha.size()[1]
    log_alpha = log_alpha.reshape(-1, n, n)

    for _ in range(num_iters):
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).reshape(-1, n, 1)
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).reshape(-1, 1, n)
        # print(_, log_alpha)
    results = torch.exp(log_alpha)
    # print("results", results.deeper(2))
    return results


def is_doubly_stochastic(mat, threshold: th.Optional[float] = 1e-4) -> torch.Tensor:
    """
    Checks if a matrix is doubly stochastic (given a threshold).
    Args:
        mat: 2D tensor (a matrix of shape [N, N])
            or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        threshold: float, the threshold for the check to pass
            if None, the maximum difference between the sum of rows and columns and 1 is returned

    Returns:
        Either a boolean tensor of shape [batch_size] (if threshold is not None)
        or a float tensor of shape [batch_size] (if threshold is None)
    """
    if threshold is None:
        # return the maximum difference between the sum of rows and columns and 1
        return torch.maximum((mat.sum(-1) - 1).abs().max(-1).values, (mat.sum(-2) - 1).abs().max(-1).values)
    return ((mat.sum(-1) - 1).abs().max(-1).values < threshold) & ((mat.sum(-2) - 1).abs().max(-1).values < threshold)


def is_permutation(mat, threshold: th.Optional[float] = 1e-4):
    """
    Checks if a matrix is a permutation matrix (given a threshold).
    By definition, a permutation matrix is a doubly stochastic matrix with
    exactly one 1 per row and column.

    To check how close a matrix is to being a permutation matrix, we check
    how close all of its entries are to 0 or 1. The maximum distance is then used
    as a measure of how close the matrix is to being a permutation matrix.

    Args:
        mat: 2D tensor (a matrix of shape [N, N])
            or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        threshold: float, the threshold for the check to pass
            if None, the maximum distance between values and 0 or 1 is returned

    Returns:
        Either a boolean tensor of shape [batch_size] (if threshold is not None)
        or a float tensor of shape [batch_size] (if threshold is None)
    """
    distance = torch.minimum((mat - 1).abs(), mat.abs())
    results = distance.max(-1).values.max(-1).values
    return results if threshold is None else results < threshold


def is_between_zero_one(mat, threshold: th.Optional[float] = 1e-4):
    """
    Checks if a matrix is between 0 and 1 (given a threshold). The difference between
    this funciton and is_permutation is that, here we only care how much of the values
    are not in between 0 and 1, while in is_permutation we only care that the values are
    close to either 0 or 1.

    is_permutation might be true (having small value)
    Args:
        mat: 2D tensor (a matrix of shape [N, N])
            or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        threshold: float, the threshold for the check to pass
            if None, the maximum distance between values and 0 or 1 is returned
    """
    # zero out all the numbers except the ones that are not between 0 and 1
    mat = torch.where((mat >= 0) & (mat <= 1.0), torch.zeros_like(mat), mat)
    distance = torch.minimum((mat - 1).abs(), mat.abs())
    results = distance.max(-1).values.max(-1).values
    return results if threshold is None else results < threshold


def evaluate_permutations(mat, threshold: th.Optional[float] = 1e-4, reduce: bool = True):
    """
    Evaluates a matrix of permutations (or a batch of matrices of permutations)

    Args:
        mat: 2D tensor (a matrix of shape [N, N]) or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
        threshold: float, the threshold for the check to pass (if None, only the distances are returned)
        reduce: whether to reduce the results into an average (boolean tensors are reduced to a proportion of True values)

    Returns:
        A dictionary with the following keys:
            doubly_stochastic_distance: the maximum difference between the sum of rows and columns and 1
            permutation_distance: the maximum distance between values and 0 or 1

    """
    results = dict(
        doubly_stochastic_distance=is_doubly_stochastic(mat, threshold=None),
        permutation_distance=is_permutation(mat, threshold=None),
        between_zero_one_distance=is_between_zero_one(mat, threshold=None),
    )
    if threshold is not None:
        results.update({f'is_{k.replace("_distance", "")}': v < threshold for k, v in results.items()})
    if reduce:
        results = {k: v.float().mean() for k, v in results.items()}
    return results


def listperm2matperm(listperm: torch.Tensor, device=None, dtype=None):
    """Converts a batch of permutations to its matricial form.
    Args:
      listperm: 2D tensor of permutations of shape [batch_size, n_objects] so that
        listperm[n] is a permutation of range(n_objects).
      device: device to place the output tensor on. (default: None)
      dtype: dtype of the output tensor. (default: None)
    Returns:
      a 3D tensor of permutations matperm of
        shape = [batch_size, n_objects, n_objects] so that matperm[n, :, :] is a
        permutation of the identity matrix, with matperm[n, i, listperm[n,i]] = 1
    """
    return torch.eye(listperm.shape[-1])[listperm.long()].to(device=device, dtype=dtype)


def hungarian(matrix_batch):
    """Solves a matching problem using the Hungarian algorithm.

    This is a wrapper for the scipy.optimize.linear_sum_assignment function. It
    solves the optimization problem max_P sum_i,j M_i,j P_i,j with P a
    permutation matrix. Notice the negative sign; the reason, the original
    function solves a minimization problem.

    Args:
        matrix_batch: A 3D tensor (a batch of matrices) with
            shape = [batch_size, N, N]. If 2D, the input is reshaped to 3D with
            batch_size = 1.
    Returns:
        listperms, a 2D integer tensor of permutations with shape [batch_size, N]
            so that listperms[n, :] is the permutation of range(N) that solves the
            problem  max_P sum_i,j M_i,j P_i,j with M = matrix_batch[n, :, :].
    """
    # keep the import here to avoid unnecessary dependency in the rest of the code
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    # perform the hungarian algorithm on the cpu
    matrix_batch = matrix_batch.detach().cpu().numpy()
    if matrix_batch.ndim == 2:
        matrix_batch = np.reshape(matrix_batch, [1, matrix_batch.shape[0], matrix_batch.shape[1]])
    sol = np.zeros((matrix_batch.shape[0], matrix_batch.shape[1]), dtype=np.int32)
    for i in range(matrix_batch.shape[0]):
        sol[i, :] = linear_sum_assignment(-matrix_batch[i, :])[1].astype(np.int32)
    return torch.from_numpy(sol)