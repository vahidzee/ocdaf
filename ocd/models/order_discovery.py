import torch
from .made import SingleMaskedBlockMADE
from torch.nn.parameter import Parameter
import typing as th
from ocd.permutation_tools import listperm2matperm, derive_deterministic_permutation, sample_permutation


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
        noise_factor: float = 0.0,
        different_noise_per_batch: bool = False,
        gamma_scaling: float = 1,
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
        p = torch.randn(n, n, requires_grad=True, device=device, dtype=dtype)

        self.noise_factor = noise_factor
        self.different_noise_per_batch = different_noise_per_batch
        self._gamma = torch.nn.Parameter(gamma_scaling * p)
        self.tau = tau
        self.n_iter = n_iter

        # the following is set for turning off the structure learning part
        self.permutation = None
        self.permutation_list = None

    @property
    def Gamma(self) -> torch.Tensor:
        return torch.linalg.qr(self._gamma)[0]

    def set_tau(self, tau: float) -> None:
        self.tau = tau

    def set_n_iter(self, n_iter: int) -> None:
        self.n_iter = n_iter

    def set_permutation(self, permutation: th.Union[th.List[int], torch.Tensor]) -> None:
        # Set the list version
        self.permutation_list = permutation if isinstance(permutation, list) else permutation.tolist()

        # Convert list into torch tensor
        if isinstance(permutation, list):
            permutation = torch.tensor(permutation, dtype=torch.int, device=self.Gamma.device)
        # Set the permanent matrix version
        self.permutation = listperm2matperm(
            torch.tensor(permutation) if isinstance(permutation, list) else permutation.unsqueeze(0)
        ).squeeze(0)
        # change the dtype of self.permutation to match the dtype of self.Gamma
        self.permutation = self.permutation.to(self.Gamma.dtype)

    def unset_permutation(self) -> None:
        self.permutation = None

    def get_permanent_matrices(self, n_sample: int, trial_and_error: int = 10) -> torch.Tensor:
        if self.permutation is None:
            # sample a set of differentiable matrices
            all_perms = sample_permutation(
                self.Gamma,
                self.noise_factor,
                n_sample * trial_and_error,
                mode="soft",
                sinkhorn_temp=self.tau,
                sinkhorn_iters=self.n_iter,
            )
            # do trial and error because sinkhorn operator is imperfect some rows might not sum to 1
            # per each sample, we will sample 'trial_and_error' amount of samples
            # and pick the one producing the maximum minimum row sum (which should be 1)
            if trial_and_error == 1:
                return all_perms

            candidate_indices = []
            for i in range(n_sample):
                perms = all_perms[i * trial_and_error : (i + 1) * trial_and_error, :, :]
                row_sums = torch.sum(perms, axis=1)
                min_row_sums = torch.min(row_sums, axis=-1)[0]
                candidate_indices.append(min_row_sums.argmax())
            # ensure that the sampled matrices are actually permanent
            return all_perms[candidate_indices, :, :]
        else:
            return self.permutation.unsqueeze(0).repeat(n_sample, 1, 1)

    def get_permutation(self) -> th.List[int]:
        """
        Returns the permutation of the input variables.

        It should not be called for training but just for checking and evaluation.
        """
        if self.permutation is not None:
            return self.permutation_list.copy()
        else:
            with torch.no_grad():
                return derive_deterministic_permutation(self.Gamma).tolist()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.different_noise_per_batch:
            P = self.get_permanent_matrices(x.shape[0])
        else:
            P = self.get_permanent_matrices(1)[0]
        self.P = P
        return self.made(x, P)
