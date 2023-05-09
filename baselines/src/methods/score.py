# Codes are adopted from the original implementation
# https://github.com/paulrolland1307/SCORE/tree/5e18c73a467428d51486d2f683349dde2607bfe1
# under the GNU Affero General Public License v3.0
# Copy right belongs to the original author https://github.com/paulrolland1307

from src.base import AbstractBaseline
import torch
import typing as th
import os
import numpy as np
import networkx as nx
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_DIR, "../../"))
from ocd.post_processing.cam_pruning import cam_pruning


def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d, d))
    for i, var in enumerate(top_order):
        A[var, top_order[i + 1 :]] = 1
    return A


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


class Score(AbstractBaseline):
    def __init__(
        self,
        dataset: th.Union["OCDDataset", str],  # type: ignore
        dataset_args: th.Optional[th.Dict[str, th.Any]] = None,
        # hyperparameters
        eta_G: float = 0.001,
        eta_H: float = 0.001,
        normalize_var: float = False,
        dispersion: th.Literal["var", "median"] = "var",
    ):
        super().__init__(dataset=dataset, dataset_args=dataset_args, name="Score")
        self.eta_G = eta_G
        self.eta_H = eta_H
        self.normalize_var = normalize_var
        self.dispersion = dispersion
        self.data = self.get_data(conversion="tensor")
        self.order = self._compute_top_order(self.data)

    def _compute_top_order(self, data):
        eta_G = self.eta_G
        eta_H = self.eta_H
        normalize_var = self.normalize_var
        dispersion = self.dispersion

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

    def estimate_order(self):
        return self.order

    def estimate_dag(self):
        dag = full_DAG(self.order)
        dag = cam_pruning(dag, np.array(self.data.detach().cpu().numpy()), cutoff=0.001)
        return nx.DiGraph(dag)
