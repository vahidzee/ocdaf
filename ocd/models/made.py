import torch
import typing as th
import functools
from .layers import OrderedBlock, AutoRegressiveDensityEstimator1D

class SingleMaskedBlockMADE(torch.nn.Module):
    def __init__(
        self,
        in_covariate_features: th.List[int],
        hidden_features_per_covariate: th.List[th.List[int]],

        # layers
        bias: bool = True,

        # Properties of the activations used in each layer
        activation: th.Optional[str] = "torch.nn.ReLU",
        activation_args: th.Optional[dict] = None,

        # Properties of the batch normalization layers
        batch_norm: bool = True,
        batch_norm_args: th.Optional[dict] = None,
        
        # general
        seed: int = 0,
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        
        # grad safety
        safe_grad_hook: str = "lambda grad: torch.where(torch.isnan(grad) + torch.isinf(grad), torch.zeros_like(grad), grad)",
    ) -> None:
        super().__init__()

        self.L = len(hidden_features_per_covariate)
        self.layers = torch.nn.Sequential(
            *[
                OrderedBlock(
                    in_cov_features = in_covariate_features if not i else hidden_features_per_covariate[i - 1],
                    out_cov_features = hidden_features_per_covariate[i],
                    bias=bias,
                    activation=activation,
                    activation_args=activation_args,
                    batch_norm=batch_norm,
                    batch_norm_args=batch_norm_args,
                    device=device,
                    dtype=dtype,
                )
                for i in range(self.L)
            ]
        )

        # TODO: finalize this
        self.density_estimator = AutoRegressiveDensityEstimator1D(
            in_cov_features=hidden_features_per_covariate[-1],
            out_cov_features=in_covariate_features,
            bias=bias,
            device=device,
            dtype=dtype,
            auto_connection=False,
        )

        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.safe_grad_hook = safe_grad_hook

    def forward(
        self,
        inputs,
        perm: th.Union[th.List[int], torch.Tensor],
    ):
        # Takes in inputs of shape (batch_size, sum of [input features per covariate]) 
        # and either a permutation of the covariates or a permanent matrix for permuting the covariates
        # If the permutation is a list, the gradients are off, otherwise, they will flow through the permutation matrix

        # The permutation matrix is a doubly stochastic matrix that is used to (soft)permute the covariates
        if isinstance(perm, list):
            n = len(perm)
            # check if perm is a permutation
            assert set(perm) == set(range(n)), "perm must be a permutation of [0, 1, ..., n-1]"
            # Create a permutation tensor from a permutation list
            P = torch.zeros((n, n))
            P[torch.arange(n), perm] = 1
            P = P.to(inputs.device)
            P = P.type(inputs.dtype)
            perm = P
        else:
            # Check if perm is a square matrix
            assert perm.shape[0] == perm.shape[1], "perm must be a square matrix"
            # check if perm is doubly stochastic
            assert torch.allclose(perm.sum(dim=0), torch.ones(perm.shape[0])), "perm must be doubly stochastic"
            assert torch.allclose(perm.sum(dim=1), torch.ones(perm.shape[0])), "perm must be doubly stochastic"
            
        results, perm = self.layers((inputs, perm))
        results = self.density_estimator(results, perm)
        
        # if results.requires_grad and safe_grad:
        #     results.register_hook(self.safe_grad_hook_function)
        return results

