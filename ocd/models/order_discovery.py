import torch
from .made import SingleMaskedBlockMADE
from torch.nn.parameter import Parameter
import typing as th
from ocd.permutation_tools import listperm2matperm, derive_deterministic_permutation


class SinkhornOrderDiscovery(torch.nn.Module):
    """ """

    def __init__(
        self,
        # MADE parameters
        in_covariate_features: th.List[int],
        hidden_features_per_covariate: th.List[th.List[int]],
        bias: bool = True,
        activation: th.Optional[str] = "torch.nn.ReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = True,
        batch_norm_args: th.Optional[dict] = None,
        # Sinkhorn parameters
        n_iter: int = 10,
        tau: float = 0.1,
        # general
        seed: int = 0,
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        safe_grad_hook: str = "lambda grad: torch.where(torch.isnan(grad) + torch.isinf(grad), torch.zeros_like(grad), grad)",
    ) -> None:
        super().__init__()
        self.in_covariate_features = in_covariate_features  # todo: cleanup
        self.made = SingleMaskedBlockMADE(
            in_covariate_features=in_covariate_features,
            hidden_features_per_covariate=hidden_features_per_covariate,
            bias=bias,
            activation=activation,
            activation_args=activation_args,
            batch_norm=batch_norm,
            batch_norm_args=batch_norm_args,
            seed=seed,
            device=device,
            dtype=dtype,
            safe_grad_hook=safe_grad_hook,
        )
        n = len(in_covariate_features)
        p = torch.zeros(n, n, requires_grad=True, device=device, dtype=dtype)
        self.Gamma = torch.nn.Parameter(p)
        self.tau = tau
        self.n_iter = n_iter

        # the following is set for turning off the structure learning part
        self.permutation = None
        self.permutation_list = None

    def set_tau(self, tau: float) -> None:
        self.tau = tau

    def set_n_iter(self, n_iter: int) -> None:
        self.n_iter = n_iter

    def set_permutation(self, permutation: th.Union[th.List[int], torch.Tensor]) -> None:
        # Set the list version
        self.permutation_list = permutation if isinstance(
            permutation, list) else permutation.tolist()

        # Convert list into torch tensor
        if isinstance(permutation, list):
            permutation = torch.tensor(
                permutation, dtype=torch.int, device=self.Gamma.device)
        # Set the permanent matrix version
        self.permutation = listperm2matperm(torch.tensor(permutation) if isinstance(
            permutation, list) else permutation.unsqueeze(0)).squeeze(0)
        # change the dtype of self.permutation to match the dtype of self.Gamma
        self.permutation = self.permutation.to(self.Gamma.dtype)

    def unset_permutation(self) -> None:
        self.permutation = None

    def get_permanent_matrix(self) -> torch.Tensor:
        return self.sinkhorn() if self.permutation is None else self.permutation

    def sinkhorn(self) -> torch.Tensor:
        """
        Sinkhorn algorithm for computing the optimal transport matrix P.
        """
        P = torch.nn.functional.logsigmoid(self.Gamma)
        P = P / self.tau
        for i in range(self.n_iter):
            P = P - torch.logsumexp(P, dim=1, keepdim=True)
            P = P - torch.logsumexp(P, dim=0, keepdim=True)
        return P.exp()

    def get_permutation(self) -> torch.Tensor:
        if self.permutation is not None:
            return self.permutation_list.copy()
        else:
            with torch.no_grad():
                return derive_deterministic_permutation(self.Gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P = self.get_permanent_matrix()
        return self.made(x, P)
