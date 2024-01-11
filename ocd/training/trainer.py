import torch
from ocd.config import (
    TrainingConfig,
    GumbelSinkhornConfig,
    GumbelTopKConfig,
    SoftSinkhornConfig,
    DataVisualizer,
    BirkhoffConfig,
)
from torch.utils.data import DataLoader
from typing import Union, Callable, Iterable, Optional
from ocd.models.oslow import OSlow
from ocd.models.permutation import hungarian

class PermutationLearningModule(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
    ):
        super().__init__()
        self.register_parameter(
            "gamma",
            torch.nn.Parameter(
                torch.randn(in_features, in_features)
            )
        )
        
class Trainer:
    def __init__(
        self, 
        model: OSlow,
        dataloader: DataLoader,
        flow_optimizer: Callable[[Iterable], torch.optim.Optimizer],
        permutation_optimizer: Callable[[Iterable], torch.optim.Optimizer],
        flow_frequency: int,
        permutation_frequency: int,
        max_epochs: int,
        flow_lr_scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler],
        permutation_lr_scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler],
        permutation_learning_config: Union[GumbelSinkhornConfig, GumbelTopKConfig, SoftSinkhornConfig],
        data_visualizer_config: Optional[DataVisualizer] = None,
        birkhoff_config: Optional[BirkhoffConfig] = None,
        device: str = "cpu",
    ):
        self.max_epochs = max_epochs
        self.model = model.to(device)
        self.permutation_learning_module = PermutationLearningModule(model.in_features).to(device)
        #= torch.nn.Parameter(
        #     torch.randn(model.in_features, model.in_features)
        # ).to(device).requires_grad_(True)
        self.dataloader = dataloader
        self.permutation_learning_config = permutation_learning_config
        self.data_visualizer_config = data_visualizer_config
        self.birkhoff_config = birkhoff_config
        self.flow_optimizer = flow_optimizer(self.model.parameters())
        self.permutation_optimizer = permutation_optimizer(self.permutation_learning_module.parameters())
        self.flow_frequency = flow_frequency
        self.permutation_frequency = permutation_frequency
        self.flow_scheduler = flow_lr_scheduler(
            self.flow_optimizer
        )
        self.permutation_scheduler = permutation_lr_scheduler(
            self.permutation_optimizer
        )

        # TODO create checkpointing
        self.checkpointing = None
        # tracking configurations
        self.data_visualizer = None
        self.brikhoff = None

    def _gumbel_top_k_train_step(self):
        pass

    def gumbel_matching(self, num_samples: int):
        gumbel_samples = -(
            -torch.rand(num_samples, self.model.d, self.model.d).log()
        ).log()

        return hungarian(self.gamma + gumbel_samples)

    def stochastic_beam_search(self, num_samples: int):
        # TODO with gamma \in R^{d}
        pass

    def daguerreotype_search(self, num_samples: int):
        # TODO with gamma \in R^{d}
        pass

    def _flow_train_step(self):
        self.gamma.requires_grad = False
        for batch in self.dataloader:
            permutations = self._get_permutation(batch.shape[0])
            self.flow_optimizer.zero_grad()
            loss = -(self.model.log_prob(batch, permutations)).mean()
            loss.backward()
            self.flow_optimizer.step()

    def train(self):
        self.model.train()

        # for epoch in range(self.max_epochs):
        #     for batch in self.dataloader:
        #         self.model.log_prob(batch, perm_mat)
