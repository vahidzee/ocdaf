# Codes are adopted from the original implementation
# https://github.com/paulrolland1307/SCORE/tree/5e18c73a467428d51486d2f683349dde2607bfe1
# under the GNU Affero General Public License v3.0
# Copy right belongs to the original author https://github.com/paulrolland1307
import numpy as np
import torch


def estimate_hessian(X, eta_G, eta_H, s=None):
    """
    Estimates the diagonal of the Hessian of log p_X at the provided samples points
    X, using first and second-order Stein identities
    """
    n, d = X.shape

    X_diff = X.unsqueeze(1) - X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2) ** 2 / (2 * s**2)) / s

    nablaK = -torch.einsum("kij,ik->kj", X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)

    nabla2K = torch.einsum("kij,ik->kj", -1 / s**2 + X_diff**2 / s**4, K)
    return -(G**2) + torch.matmul(torch.inverse(K + eta_H * torch.eye(n)), nabla2K)


def fit(data, eta_G=0.001, eta_H=0.001, normalize_var=False, dispersion="var"):
    n, d = data.shape
    order = []
    active_nodes = list(range(d))
    for i in range(d - 1):
        H = estimate_hessian(data, eta_G, eta_H)
        if normalize_var:
            H = H / H.mean(axis=0)
        if dispersion == "var":  # The one mentioned in the paper
            l = int(H.var(axis=0).argmin())
        elif dispersion == "median":
            med = H.median(axis=0)[0]
            l = int((H - med).abs().mean(axis=0).argmin())
        else:
            raise Exception("Unknown dispersion criterion")
        order.append(active_nodes[l])
        active_nodes.pop(l)
        data = torch.hstack([data[:, 0:l], data[:, l + 1 :]])
    order.append(active_nodes[0])
    order.reverse()
    return order
