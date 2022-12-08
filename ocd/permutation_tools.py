"""
This file contains the code for computing the optimal transport matrix P
using the soft Sinkhorn formulation as well as using the hungarian algorithm
"""

import torch
import typing as th
import numpy as np
from scipy.optimize import linear_sum_assignment


def sinkhorn(log_alpha, n_iters=20):
    # Code from https://github.com/sharpenb/Differentiable-DAG-sampling
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the succesive row and column
    normalization.
    -To ensure positivity, the effective input to sinkhorn has to be
    exp(log_alpha) (elementwise).
    -However, for stability, sinkhorn works in the log-space. It is only at
    return time that entries are exponentiated.
    [1] Sinkhorn, Richard and Knopp, Paul.
    Concerning nonnegative matrices and doubly stochastic
    matrices. Pacific Journal of Mathematics, 1967
    Args:
    log_alpha: 2D tensor (a matrix of shape [N, N])
        or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
    n_iters: number of sinkhorn iterations (in practice, as little as 20
        iterations are needed to achieve decent convergence for N~100)
    Returns:
    A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
        converted to 3D tensors with batch_size equals to 1)
    """
    n = log_alpha.size()[1]
    log_alpha = log_alpha.reshape(-1, n, n)

    for _ in range(n_iters):
        log_alpha = log_alpha - \
            (torch.logsumexp(log_alpha, dim=2, keepdim=True)).reshape(-1, n, 1)
        log_alpha = log_alpha - \
            (torch.logsumexp(log_alpha, dim=1, keepdim=True)).reshape(-1, 1, n)
    return torch.exp(log_alpha)


def sample_gumbel(shape, device, eps=1e-20):
    # Code from https://github.com/sharpenb/Differentiable-DAG-sampling
    """Samples arbitrary-shaped standard gumbel variables.
    Args:
    shape: list of integers
    eps: float, for numerical stability
    Returns:
    A sample of standard Gumbel random variables
    """

    u = torch.rand(shape, device=device).float()
    return -torch.log(-torch.log(u + eps) + eps)


def matching(matrix_batch):
    # Code from https://github.com/sharpenb/Differentiable-DAG-sampling
    """Solves a matching problem for a batch of matrices.
    This is a wrapper for the scipy.optimize.linear_sum_assignment function. It
    solves the optimization problem max_P sum_i,j M_i,j P_i,j with P a
    permutation matrix. Notice the negative sign; the reason, the original
    function solves a minimization problem
    Args:
    matrix_batch: A 3D tensor (a batch of matrices) with
        shape = [batch_size, N, N]. If 2D, the input is reshaped to 3D with
        batch_size = 1.
    Returns:
    listperms, a 2D integer tensor of permutations with shape [batch_size, N]
        so that listperms[n, :] is the permutation of range(N) that solves the
        problem  max_P sum_i,j M_i,j P_i,j with M = matrix_batch[n, :, :].
    """

    def hungarian(x):
        if x.ndim == 2:
            x = np.reshape(x, [1, x.shape[0], x.shape[1]])
        sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
        for i in range(x.shape[0]):
            sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
        return sol

    listperms = hungarian(matrix_batch.detach().cpu().numpy())
    listperms = torch.from_numpy(listperms)
    return listperms


def gumbel_sinkhorn(log_alpha,
                    temp=1.0, n_samples=1, noise_factor=1.0, n_iters=20,
                    squeeze=True, hard=False):
    # Code from https://github.com/sharpenb/Differentiable-DAG-sampling
    """Random doubly-stochastic matrices via gumbel noise.
    In the zero-temperature limit sinkhorn(log_alpha/temp) approaches
    a permutation matrix. Therefore, for low temperatures this method can be
    seen as an approximate sampling of permutation matrices, where the
    distribution is parameterized by the matrix log_alpha
    The deterministic case (noise_factor=0) is also interesting: it can be
    shown that lim t->0 sinkhorn(log_alpha/t) = M, where M is a
    permutation matrix, the solution of the
    matching problem M=arg max_M sum_i,j log_alpha_i,j M_i,j.
    Therefore, the deterministic limit case of gumbel_sinkhorn can be seen
    as approximate solving of a matching problem, otherwise solved via the
    Hungarian algorithm.
    Warning: the convergence holds true in the limit case n_iters = infty.
    Unfortunately, in practice n_iter is finite which can lead to numerical
    instabilities, mostly if temp is very low. Those manifest as
    pseudo-convergence or some row-columns to fractional entries (e.g.
    a row having two entries with 0.5, instead of a single 1.0)
    To minimize those effects, try increasing n_iter for decreased temp.
    On the other hand, too-low temperature usually lead to high-variance in
    gradients, so better not choose too low temperatures.
    Args:
    log_alpha: 2D tensor (a matrix of shape [N, N])
        or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
    temp: temperature parameter, a float.
    n_samples: number of samples
    noise_factor: scaling factor for the gumbel samples. Mostly to explore
        different degrees of randomness (and the absence of randomness, with
        noise_factor=0)
    n_iters: number of sinkhorn iterations. Should be chosen carefully, in
        inverse corresponde with temp to avoid numerical stabilities.
    squeeze: a boolean, if True and there is a single sample, the output will
        remain being a 3D tensor.
    hard: boolean
    Returns:
    sink: a 4D tensor of [batch_size, n_samples, N, N] i.e.
        batch_size *n_samples doubly-stochastic matrices. If n_samples = 1 and
        squeeze = True then the output is 3D.
    log_alpha_w_noise: a 4D tensor of [batch_size, n_samples, N, N] of
        noisy samples of log_alpha, divided by the temperature parameter. If
        n_samples = 1 then the output is 3D.
    """
    n = log_alpha.size()[1]
    log_alpha = log_alpha.reshape(-1, n, n)
    batch_size = log_alpha.size()[0]
    log_alpha_w_noise = log_alpha.repeat(n_samples, 1, 1)

    if noise_factor == 0:
        noise = 0.0
    else:
        noise = sample_gumbel([n_samples * batch_size, n, n]) * noise_factor

    log_alpha_w_noise = log_alpha_w_noise + noise
    log_alpha_w_noise = log_alpha_w_noise / temp

    log_alpha_w_noise_copy = log_alpha_w_noise.clone()
    sink = sinkhorn(log_alpha_w_noise_copy, n_iters)

    if n_samples > 1 or squeeze is False:
        sink = sink.reshape(n_samples, batch_size, n, n)
        sink = torch.transpose(sink, 1, 0)
        log_alpha_w_noise = log_alpha_w_noise.reshape(
            n_samples, batch_size, n, n)
        log_alpha_w_noise = torch.transpose(log_alpha_w_noise, 1, 0)

    ret = (sink, log_alpha_w_noise)

    if hard:
        # Straight through.
        log_alpha_w_noise_flat = torch.transpose(log_alpha_w_noise, 0, 1)
        log_alpha_w_noise_flat = log_alpha_w_noise_flat.view(-1, n, n)
        hard_perms_inf = matching(log_alpha_w_noise_flat)
        inverse_hard_perms_inf = invert_listperm(hard_perms_inf)
        sink_hard = listperm2matperm(hard_perms_inf).to(device).float()
        ret = (sink_hard - sink.detach() + sink, log_alpha_w_noise)

    return ret


def listperm2matperm(listperm):
    """Converts a batch of permutations to its matricial form.
    Args:
    listperm: 2D tensor of permutations of shape [batch_size, n_objects] so that
      listperm[n] is a permutation of range(n_objects).
    Returns:
    a 3D tensor of permutations matperm of
      shape = [batch_size, n_objects, n_objects] so that matperm[n, :, :] is a
      permutation of the identity matrix, with matperm[n, i, listperm[n,i]] = 1
    """
    n_objects = listperm.size()[1]
    eye = np.eye(n_objects)[listperm]
    eye = torch.tensor(eye, dtype=torch.int32)
    return eye


def derive_deterministic_permutation(permutation_table: torch.Tensor, output_matrix: bool = False):
    """
    Args:
        permutation_table: [batch_size, n, n] or [n, n]
        output_matrix: if True, returns a matrix representation of permutations
    Returns:
        permutation: [batch_size, n] or [n]
    """
    return sample_permutation(permutation_table, noise_factor=0, n_samples=1, mode='hard', output_matrix=output_matrix)[0]


def sample_permutation(permutation_table: torch.Tensor, noise_factor: float, n_samples: int, mode: str,
                       sinkhorn_temp: th.Optional[float] = None, sinkhorn_iters: th.Optional[int] = None,
                       output_matrix: bool = True):
    """
    Args:
        n_samples: number of samples
        mode: 'hard' or 'soft'
        permutation_table: [batch_size, n, n] or [n, n]
        noise_factor: scaling factor for the gumbel samples. Mostly to explore 
            different degrees of randomness (and the absence of randomness, with
            noise_factor=0)
        sinkhorn_temp: temperature parameter, a float.
        sinkhorn_iters: number of sinkhorn iterations. Should be chosen carefully, in
            inverse corresponde with temp to avoid numerical stabilities.
        output_matrix: if True, returns a matrix representation of permutations, otherwise returns
            a list representation.


    Returns:
        P: [n_samples, n, n]
    """
    permutation_table = torch.nn.functional.logsigmoid(permutation_table)
    n = permutation_table.shape[-1]
    permutation_table = permutation_table.reshape(-1, n, n)
    batch_size = permutation_table.shape[0]
    permutation_table = permutation_table.repeat(n_samples, 1, 1)

    if noise_factor == 0:
        noise = 0.0
    else:
        noise = sample_gumbel([n_samples * batch_size, n, n],
                              device=permutation_table.device) * noise_factor

    permutation_table = permutation_table + noise

    # print(permutation_table)

    if mode == 'hard':
        perms = matching(permutation_table)
        if output_matrix:
            perms = listperm2matperm(perms)
    elif mode == 'soft':
        if sinkhorn_temp is None:
            raise NotImplementedError(
                "sinkhorn_temp must be specified for soft mode")
        if sinkhorn_iters is None:
            raise NotImplementedError(
                "sinkhorn_iters must be specified for soft mode")
        perms = sinkhorn(permutation_table / sinkhorn_temp, sinkhorn_iters)
        if output_matrix is False:
            perms = perms.argmax(-1)
    else:
        raise NotImplementedError("mode must be either 'hard' or 'soft'")

    perms = perms.reshape([n_samples, batch_size] + list(perms.shape[1:]))
    perms = torch.transpose(perms, 1, 0)
    return perms.squeeze(0)
