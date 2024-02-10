import torch
import abc

from ocd.models import OSlow
from ocd.training.utils import sample_gumbel_noise, hungarian, turn_into_matrix
from typing import Optional, Literal


def _sinkhorn(log_x: torch.Tensor, iters: int, temp: float):
    """
    Perform a sequence of normalizations to make the matrix doubly stochastic
    This is known as the Sinkhorn operator.
    When the number of iterations are large, this value converges to the Hungarian algorithm's result.
    """
    n = log_x.size()[1]
    log_x = log_x.reshape(-1, n, n) / temp
    for _ in range(iters):
        log_x = log_x - (torch.logsumexp(log_x, dim=2, keepdim=True)).reshape(-1, n, 1)
        log_x = log_x - (torch.logsumexp(log_x, dim=1, keepdim=True)).reshape(-1, 1, n)
    results = torch.exp(log_x)
    return results


class PermutationLearningModule(torch.nn.Module, abc.ABC):
    def __init__(
        self,
        in_features: int,
    ):
        super().__init__()
        self.in_features = in_features

    @property
    def gamma(self):
        return self._gamma

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        raise NotImplementedError

    def flow_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        raise NotImplementedError


class SoftSort(PermutationLearningModule):
    def __init__(self, in_features: int, *args, temp: float = 0.1, **kwargs):
        super().__init__(in_features=in_features, *args, **kwargs)
        self.register_parameter("_gamma", torch.nn.Parameter(torch.randn(in_features)))
        self.temp = temp

    def sample_hard_permutations(
        self,
        num_samples: int,
        return_noises: bool = False,
        unique_and_resample: bool = False,
        gumbel_std: float = 1.0,
    ):
        gumbel_noise = sample_gumbel_noise(
            (num_samples, *self.gamma.shape),
            device=self.gamma.device,
            std=gumbel_std,
        )
        scores = self.gamma + gumbel_noise
        # perform argsort on every line of scores
        permutations = torch.argsort(scores, dim=-1)
        if unique_and_resample:
            permutations = torch.unique(permutations, dim=0)
            permutations = permutations[
                torch.randint(
                    0,
                    permutations.shape[0],
                    (num_samples,),
                    device=permutations.device,
                )
            ]

        # turn permutations into permutation matrices
        ret = torch.stack([turn_into_matrix(perm) for perm in permutations])

        # add some random noise to the permutation matrices
        if return_noises:
            return ret, gumbel_noise
        return ret

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        permutations, gumbel_noise = self.sample_hard_permutations(
            batch.shape[0], return_noises=True, gumbel_std=temperature
        )
        scores = self.gamma + gumbel_noise
        all_ones = torch.ones_like(scores)
        scores_sorted = torch.sort(scores, dim=-1).values
        logits = (
            -(
                (
                    scores_sorted.unsqueeze(-1)
                    @ all_ones.unsqueeze(-1).transpose(-1, -2)
                    - all_ones.unsqueeze(-1) @ scores.unsqueeze(-1).transpose(-1, -2)
                )
                ** 2
            )
            / self.temp
        )
        # perform a softmax on the last dimension of logits
        soft_permutations = torch.softmax(logits, dim=-1)
        log_probs = model.log_prob(
            batch,
            perm_mat=(permutations - soft_permutations).detach() + soft_permutations,
        )
        return -log_probs.mean()

    def flow_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        permutations = self.sample_hard_permutations(
            batch.shape[0], unique_and_resample=True, gumbel_std=temperature
        ).detach()
        log_probs = model.log_prob(batch, perm_mat=permutations)
        return -log_probs.mean()


class PermutationMatrixLearningModule(PermutationLearningModule, abc.ABC):
    def __init__(
        self,
        in_features: int,
        *args,
        **kwargs,
    ):
        super().__init__(in_features, *args, **kwargs)
        self.register_parameter(
            "_gamma", torch.nn.Parameter(torch.randn(in_features, in_features))
        )

    def sample_hard_permutations(
        self,
        num_samples: int,
        return_noises: bool = False,
        unique_and_resample: bool = False,
        gumbel_std: float = 1.0,
    ):
        gumbel_noise = sample_gumbel_noise(
            (num_samples, *self.gamma.shape),
            device=self.gamma.device,
            std=gumbel_std,
        )
        permutations = hungarian(self.gamma + gumbel_noise).to(self.gamma.device)
        if unique_and_resample:
            permutations = torch.unique(permutations, dim=0)
            permutations = permutations[
                torch.randint(
                    0,
                    permutations.shape[0],
                    (num_samples,),
                    device=permutations.device,
                )
            ]
        # turn permutations into permutation matrices
        ret = torch.stack([turn_into_matrix(perm) for perm in permutations])
        # add some random noise to the permutation matrices
        if return_noises:
            return ret, gumbel_noise
        return ret

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        raise NotImplementedError

    def flow_learning_loss(
        self, model: OSlow, batch: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        permutations = self.sample_hard_permutations(
            batch.shape[0], unique_and_resample=True, gumbel_std=temperature
        ).detach()
        log_probs = model.log_prob(batch, perm_mat=permutations)
        return -log_probs.mean()


class SoftSinkhorn(PermutationMatrixLearningModule):
    def __init__(
        self, in_features: int, *args, temp: float = 0.1, iters: int = 20, **kwargs
    ):
        super().__init__(in_features, *args, **kwargs)
        self.temp = temp
        self.iters = iters

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        _, gumbel_noise = self.sample_hard_permutations(
            batch.shape[0], return_noises=True, gumbel_std=temperature
        )
        soft_permutations = _sinkhorn(
            self.gamma + gumbel_noise, iters=self.iters, temp=self.temp
        )
        log_probs = model.log_prob(batch, perm_mat=soft_permutations)
        return -log_probs.mean()


class GumbelSinkhornStraightThrough(PermutationMatrixLearningModule):
    def __init__(
        self,
        in_features: int,
        *args,
        temp: float = 0.1,
        iters: int = 20,
        **kwargs,
    ):
        super().__init__(in_features, *args, **kwargs)
        self.temp = temp
        self.iters = iters

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        permutations, gumbel_noise = self.sample_hard_permutations(
            batch.shape[0], return_noises=True, gumbel_std=temperature
        )
        soft_permutations = _sinkhorn(
            self.gamma + gumbel_noise, iters=self.iters, temp=self.temp
        )
        log_probs = model.log_prob(
            batch,
            perm_mat=(permutations - soft_permutations).detach() + soft_permutations,
        )
        return -log_probs.mean()


class GumbelTopK(PermutationMatrixLearningModule):
    def __init__(
        self,
        in_features: int,
        num_samples: int,
        *args,
        different_flow_loss: bool = False,
        **kwargs,
    ):
        super().__init__(in_features, *args, **kwargs)
        self.num_samples = num_samples
        self.different_flow_loss = different_flow_loss

    def _loss(
        self, model: OSlow, batch: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        permutations = self.sample_hard_permutations(
            self.num_samples, gumbel_std=temperature
        )
        unique_perms = torch.unique(permutations, dim=0)
        b_size = batch.shape[0]
        n_unique = unique_perms.shape[0]

        # shape: (num_uniques, )
        scores = torch.sum(
            unique_perms.reshape(unique_perms.shape[0], -1) * self.gamma.reshape(1, -1),
            dim=-1,
        )

        unique_perms = unique_perms.repeat_interleave(b_size, dim=0)
        batch = batch.repeat(n_unique, 1)  # shape: (batch * num_uniques, d)

        log_probs = model.log_prob(batch, perm_mat=unique_perms)
        log_probs = log_probs.reshape(n_unique, b_size)  # shape: (batch, num_uniques, )
        losses = -log_probs.mean(axis=-1)  # shape: (num_uniques, )

        return torch.softmax(scores, dim=0) @ losses

    def permutation_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        return self._loss(batch=batch, model=model, temperature=temperature)

    def flow_learning_loss(
        self, batch: torch.Tensor, model: OSlow, temperature: float = 1.0
    ) -> torch.Tensor:
        if self.different_flow_loss:
            super().flow_learning_loss(
                batch=batch, model=model, temperature=temperature
            )
        return self._loss(batch=batch, model=model, temperature=temperature)


class ContrastiveDivergence(PermutationMatrixLearningModule):
    def __init__(
        self,
        in_features: int,
        num_samples: int,
        *args,
        **kwargs,
    ):
        super().__init__(in_features, *args, **kwargs)
        self.num_samples = num_samples

    def permutation_learning_loss(
        self, model: OSlow, batch: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        with torch.no_grad():
            permutations = self.sample_hard_permutations(
                self.num_samples, gumbel_std=temperature
            ).detach()
            unique_perms, counts = torch.unique(permutations, dim=0, return_counts=True)
            permutations_repeated = torch.repeat_interleave(
                unique_perms, batch.shape[0], dim=0
            )
            batch_repeated = batch.repeat(len(unique_perms), 1)

            scores = model.log_prob(batch_repeated, perm_mat=permutations_repeated)
            scores = scores.reshape(len(unique_perms), -1).mean(dim=-1)

        all_energies = torch.einsum("ijk,jk->i", unique_perms, self.gamma)
        weight_free_term = torch.sum(all_energies * counts) / torch.sum(counts)
        return -torch.sum(
            scores * (all_energies - weight_free_term) * counts
        ) / torch.sum(counts)
