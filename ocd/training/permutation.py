import torch
import abc

from ocd.models import OSlow
from ocd.training.utils import sample_gumbel_noise, hungarian, turn_into_matrix
from typing import Optional

class PermutationLearningModule(torch.nn.Module, abc.ABC):
    def __init__(self, in_features: int, parameterization_type: Literal['sigmoid', 'vanilla'] = 'sigmoid'):
        super().__init__()
        self.register_parameter(
            "_gamma", torch.nn.Parameter(torch.randn(in_features, in_features))
        )
        self.parameterization_type = parameterization_type

    @property
    def gamma(self):
        if self.parameterization_type == 'sigmoid':
            return torch.sigmoid(self._gamma)
        return self._gamma
    
    def sample_hard_permutations(self, num_samples: int):
        gumbel_noise = sample_gumbel_noise(
            (num_samples, *self.gamma.shape), device=self.gamma.device
        )
        permutations = hungarian(self.gamma + gumbel_noise).to(self.gamma.device)
        # turn permutations into permutation matrices
        ret = torch.stack([turn_into_matrix(perm) for perm in permutations])
        # add some random noise to the permutation matrices
        return ret

    def permutation_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        raise NotImplementedError
    
    def flow_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        raise NotImplementedError


class GumbelTopK(PermutationLearningModule):
    def __init__(self, in_features: int, num_samples: int, sampling_method: str):
        super().__init__(in_features)
        self.num_samples = num_samples

    def _loss(self, model: OSlow, batch: torch.Tensor) -> torch.Tensor:
        permutations = self.sample_hard_permutations(self.num_samples)
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

        log_probs = model.log_prob(
            batch, perm_mat=unique_perms
        )  # shape: (batch * num_uniques, )

        log_probs = log_probs.reshape(n_unique, b_size)
        losses = -log_probs.mean(axis=-1)  # shape: (num_uniques, )

        return torch.exp(scores) @ losses / torch.exp(scores).sum()

    def permutation_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        return self._loss(batch, model)
    
    def flow_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        return self._loss(batch, model)
    
    
class GumbelSinkhorn(PermutationLearningModule):
    
    def __init__(self, in_features: int, num_samples: int, *args, **kwargs):
        super().__init__(in_features)  
        raise NotImplementedError("Not implemented yet")


class ContrastiveDivergence(PermutationLearningModule):
    def __init__(self, in_features: int, num_samples: int, processing_batch_size: Optional[int] = None):
        super().__init__(in_features, parameterization_type='vanilla')
        self.num_samples = num_samples
        self.processing_batch_size = processing_batch_size = processing_batch_size or num_samples

    def permutation_learning_loss(self, model: OSlow, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            permutations = self.sample_hard_permutations(self.num_samples).detach()
            
            permutations_repeated = torch.repeat_interleave(permutations, batch.shape[0], dim=0)
            batch_repeated = batch.repeat(self.num_samples, 1)
            
            scores = []
            for batch_chunk, permutations_chunk in zip(
                torch.split(batch_repeated, self.processing_batch_size),
                torch.split(permutations_repeated, self.processing_batch_size),
            ):
                scores.append(model.log_prob(batch_chunk, perm_mat=permutations_chunk).detach())
            scores = torch.cat(scores)
            scores = scores.reshape(self.num_samples, -1).mean(dim=-1)
        
        all_energies =  torch.einsum("ijk,jk->i", permutations, self.gamma)
        weight_free_term = all_energies.mean()
        return -(scores * (all_energies - weight_free_term)).mean()
    
    def flow_learning_loss(self, model: OSlow, batch: torch.Tensor) -> torch.Tensor:
        permutations = self.sample_hard_permutations(batch.shape[0]).detach()
        log_probs = model.log_prob(batch, perm_mat=permutations)
        return -log_probs.mean()