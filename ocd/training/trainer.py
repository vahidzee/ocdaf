import torch
from ocd.config import (
    TrainingConfig,
    GumbelSinkhornConfig,
    GumbelTopKConfig,
    SoftSinkhornConfig,
)
from torch.utils.data import DataLoader
from typing import Union

from ocd.models.permutation import hungarian


class Trainer:
    def __init__(
        self, config: TrainingConfig, model: torch.nn.Module, dataloader: DataLoader
    ):
        self.config = config
        self.model = model
        self.gamma = torch.nn.Parameter(
            torch.randn(model.d, model.d), device=config.device
        )
        self.dataloader = dataloader

        self.flow_optimizer = self.config.flow_optimizer(self.model.parameters)
        self.permutation_optimizer = self.config.permutation_optimizer(self.gamma)

        self.flow_scheduler = self.config.scheduler.flow_lr_scheduler(
            self.flow_optimizer
        )
        self.permutation_scheduler = self.config.scheduler.permutation_lr_scheduler(
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
        self.model.to(self.config.device)
        self.model.train()

        for epoch in range(self.config.max_epochs):
            for batch in self.dataloader:
                self.model.log_prob(batch, perm_mat)
