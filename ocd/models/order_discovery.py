
import torch
from .made import SingleMaskedBlockMADE
from torch.nn.parameter import Parameter
import typing as th
from .sinkhorn import sinkhorn

class SinkhornOrderDiscovery(torch.nn.Module):
    """
    
    """

    def __init__(self,
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
        self.Gamma = Parameter(torch.randn(n, n, device=device, dtype=dtype))
        self.tau = tau
        self.n_iter = n_iter
    
    def set_tau(self, tau: float) -> None:
        self.tau = tau
    
    def set_n_iter(self, n_iter: int) -> None:
        self.n_iter = n_iter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        P = sinkhorn(self.Gamma, self.tau, self.n_iter)
        return self.made(x, P)