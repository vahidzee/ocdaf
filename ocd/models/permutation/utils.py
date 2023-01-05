import torch

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
    n = log_alpha.size()[1]
    log_alpha = log_alpha.reshape(-1, n, n)

    for _ in range(num_iters):
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).reshape(-1, n, 1)
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).reshape(-1, 1, n)
    return torch.exp(log_alpha)


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
