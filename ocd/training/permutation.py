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
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        
    def permutation_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        raise NotImplementedError
    
    def flow_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        raise NotImplementedError


class SoftSort(PermutationLearningModule):
    def __init__(self, in_features: int, temp: float = 0.1, parameterization_type: Literal['sigmoid', 'vanilla'] = 'sigmoid'):
        super().__init__(in_features=in_features)
        self.register_parameter(
            "_gamma", torch.nn.Parameter(torch.randn(in_features))
        )
        self.parameterization_type = parameterization_type
        self.temp = temp
    
    @property
    def gamma(self):
        if self.parameterization_type == 'sigmoid':
            return torch.sigmoid(self._gamma)
        elif self.parameterization_type == 'vanilla':
            return self._gamma
        else:
            raise ValueError(f"Unknown parameterization type: {self.parameterization_type}")
        
    def sample_hard_permutations(self, num_samples: int, return_noises: bool = False):
        gumbel_noise = sample_gumbel_noise(
            (num_samples, *self.gamma.shape), device=self.gamma.device
        )
        scores = self.gamma + gumbel_noise
        # perform argsort on every line of scores
        permutations = torch.argsort(scores, dim=-1)
        # turn permutations into permutation matrices
        ret = torch.stack([turn_into_matrix(perm) for perm in permutations])
        
        # add some random noise to the permutation matrices
        if return_noises:
            return ret, gumbel_noise
        return ret

    def permutation_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        permutations, gumbel_noise = self.sample_hard_permutations(batch.shape[0], return_noises=True)
        scores = self.gamma + gumbel_noise
        all_ones = torch.ones_like(scores)
        scores_sorted = torch.sort(scores, dim=-1).values
        logits = -(scores_sorted.unsqueeze(-1) @ all_ones.unsqueeze(-1).transpose(-1, -2) - all_ones.unsqueeze(-1) @ scores.unsqueeze(-1).transpose(-1, -2))**2 / self.temp
        # perform a softmax on the last dimension of logits
        soft_permutations = torch.softmax(logits, dim=-1)
        log_probs = model.log_prob(batch, perm_mat=(permutations-soft_permutations).detach() + soft_permutations)
        return -log_probs.mean()
    
    def flow_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        permutations = self.sample_hard_permutations(batch.shape[0]).detach()
        log_probs = model.log_prob(batch, perm_mat=permutations)
        return -log_probs.mean()
    

class PermutationMatrixLearningModule(torch.nn.Module, abc.ABC): 
    def __init__(self, in_features: int, parameterization_type: Literal['sigmoid', 'vanilla', 'log_sigmoid'] = 'sigmoid'):
        super().__init__()
        self.register_parameter(
            "_gamma", torch.nn.Parameter(torch.randn(in_features, in_features))
        )
        self.parameterization_type = parameterization_type

    @property
    def gamma(self):
        if self.parameterization_type == 'sigmoid':
            return torch.sigmoid(self._gamma)
        elif self.parameterization_type == 'log_sigmoid':
            return torch.log(1 + torch.exp(self._gamma))
        elif self.parameterization_type == 'vanilla':
            return self._gamma
        else:
            raise ValueError(f"Unknown parameterization type: {self.parameterization_type}")
    
    def sample_hard_permutations(self, num_samples: int, return_noises: bool = False):
        gumbel_noise = sample_gumbel_noise(
            (num_samples, *self.gamma.shape), device=self.gamma.device
        )
        permutations = hungarian(self.gamma + gumbel_noise).to(self.gamma.device)
        # turn permutations into permutation matrices
        ret = torch.stack([turn_into_matrix(perm) for perm in permutations])
        # add some random noise to the permutation matrices
        if return_noises:
            return ret, gumbel_noise
        return ret

    def permutation_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        raise NotImplementedError
    
    def flow_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        raise NotImplementedError
    
class GumbelSinkhornStraightThrough(PermutationMatrixLearningModule):
    def __init__(
        self, 
        in_features: int, 
        temp: float = 0.1,
        iters: int = 20,
    ):
        super().__init__(in_features)  
        self.temp = temp
        self.iters = iters

    def permutation_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        permutations, gumbel_noise = self.sample_hard_permutations(batch.shape[0], return_noises=True)
        soft_permutations = _sinkhorn(self.gamma + gumbel_noise, iters=self.iters, temp=self.temp)
        log_probs = model.log_prob(batch, perm_mat=(permutations-soft_permutations).detach() + soft_permutations)
        return -log_probs.mean()

    def flow_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        permutations = self.sample_hard_permutations(batch.shape[0]).detach()
        log_probs = model.log_prob(batch, perm_mat=permutations)
        return -log_probs.mean()
    
    
class GumbelTopK(PermutationMatrixLearningModule):
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
        return self._loss(batch=batch, model=model)
    
    def flow_learning_loss(self, batch: torch.Tensor, model: OSlow) -> torch.Tensor:
        return self._loss(batch=batch, model=model)
    


class ContrastiveDivergence(PermutationMatrixLearningModule):
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