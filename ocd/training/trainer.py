import torch

from ocd.config import (
    GumbelSinkhornStraightThroughConfig,
    GumbelTopKConfig,
    SoftSinkhornConfig,
    ContrastiveDivergenceConfig,
    SoftSortConfig,
    BirkhoffConfig,
)

import wandb
import matplotlib.pyplot as plt
import networkx as nx

from torch.utils.data import DataLoader
from typing import Union, Callable, Iterable, Optional
from ocd.models.oslow import OSlow
from ocd.training.permutation import (
    GumbelTopK,
    ContrastiveDivergence,
    GumbelSinkhornStraightThrough,
    SoftSort,
    SoftSinkhorn,
)
from ocd.visualization.birkhoff import visualize_birkhoff_polytope
from ocd.evaluation import count_backward

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Trainer:
    def __init__(
        self,
        model: OSlow,
        dag: nx.DiGraph,
        flow_dataloader: DataLoader,
        perm_dataloader: DataLoader,
        flow_optimizer: Callable[[Iterable], torch.optim.Optimizer],
        permutation_optimizer: Callable[[Iterable], torch.optim.Optimizer],
        flow_frequency: int,
        permutation_frequency: int,
        max_epochs: int,
        flow_lr_scheduler: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ],
        permutation_lr_scheduler: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ],
        permutation_learning_config: Union[
            GumbelSinkhornStraightThroughConfig,
            GumbelTopKConfig,
            SoftSinkhornConfig,
            ContrastiveDivergenceConfig,
        ],
        birkhoff_config: Optional[BirkhoffConfig] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.model = model.to(device)
        permutation_learning_kwargs = permutation_learning_config.model_dump()
        method = permutation_learning_kwargs.pop("method")
        if method == "contrastive-divergence":
            self.permutation_learning_module = ContrastiveDivergence(
                model.in_features, **permutation_learning_kwargs
            ).to(device)
        elif method == "gumbel-top-k":
            self.permutation_learning_module = GumbelTopK(
                model.in_features, **permutation_learning_kwargs
            ).to(device)
        elif method == "straight-through-sinkhorn":
            self.permutation_learning_module = GumbelSinkhornStraightThrough(
                model.in_features, **permutation_learning_kwargs
            ).to(device)
        elif method == "soft-sort":
            self.permutation_learning_module = SoftSort(
                model.in_features, **permutation_learning_kwargs
            ).to(device)
        elif method == "soft-sinkhorn":
            self.permutation_learning_module = SoftSinkhorn(
                model.in_features, **permutation_learning_kwargs
            ).to(device)
        else:
            # TODO: update and add other baselines for ablation study
            raise ValueError(
                "permutation_learning_config must be of type GumbelSinkhornStraightThroughConfig or ContrastiveDivergenceConfig"
            )

        self.flow_dataloader = flow_dataloader
        self.perm_dataloader = perm_dataloader
        self.permutation_learning_config = permutation_learning_config
        self.birkhoff_config = birkhoff_config
        self.flow_optimizer = flow_optimizer(self.model.parameters())
        self.permutation_optimizer = permutation_optimizer(
            self.permutation_learning_module.parameters()
        )
        self.flow_frequency = flow_frequency
        self.permutation_frequency = permutation_frequency
        self.flow_scheduler = flow_lr_scheduler(self.flow_optimizer)
        self.permutation_scheduler = permutation_lr_scheduler(
            self.permutation_optimizer
        )
        self.dag = dag

        self.perm_step_count = 0
        self.flow_step_count = 0

        # TODO create checkpointing
        self.checkpointing = None

    def flow_train_step(self):
        self.permutation_learning_module._gamma.requires_grad = False
        for batch in self.flow_dataloader:
            batch = batch.to(self.model.device)
            self.flow_optimizer.zero_grad()
            loss = self.permutation_learning_module.flow_learning_loss(
                model=self.model, batch=batch
            )
            loss.backward()
            self.flow_optimizer.step()
            self.flow_step_count += 1
            wandb.log({"flow/step": self.flow_step_count})
            wandb.log({"flow/loss": loss.item()})
        self.permutation_learning_module._gamma.requires_grad = True
        return loss

    def permutation_train_step(self):
        # stop gradient model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for batch in self.perm_dataloader:
            batch = batch.to(self.model.device)
            self.permutation_optimizer.zero_grad()
            loss = self.permutation_learning_module.permutation_learning_loss(
                model=self.model, batch=batch
            )
            loss.backward()
            self.permutation_optimizer.step()
            self.perm_step_count += 1
            wandb.log({"permutation/step": self.perm_step_count})
            wandb.log({"permutation/loss": loss.item()})
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        return loss

    def train(self):
        self.model.train()
        self.permutation_learning_module.train()

        true_epochs = (
            self.max_epochs // (self.flow_frequency + self.permutation_frequency) + 1
        )
        for epoch in range(true_epochs):
            for i in range(self.flow_frequency):
                loss = self.flow_train_step()
                logging.info(
                    f"Flow step {epoch * self.flow_frequency + i} / {true_epochs * self.flow_frequency}, flow loss: {loss.item()}"
                )
                if isinstance(
                    self.flow_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.flow_scheduler.step(loss.item())
                else:
                    self.flow_scheduler.step()

            for j in range(self.permutation_frequency):
                loss = self.permutation_train_step()

                logging.info(
                    f"Permutation step {epoch * self.permutation_frequency + i} / {true_epochs * self.permutation_frequency}, permutation loss: {loss.item()}"
                )
                if isinstance(
                    self.permutation_scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.permutation_scheduler.step(loss.item())
                else:
                    self.permutation_scheduler.step()

                # log the evaluation metrics
                permutation_samples = (
                    self.permutation_learning_module.sample_hard_permutations(100)
                )
                # find the majority of permutations being sampled
                permutation, counts = torch.unique(
                    permutation_samples, dim=0, return_counts=True
                )
                # find the permutation with the highest count
                permutation = permutation[counts.argmax()]
                permutation = permutation.argmax(dim=-1).cpu().numpy().tolist()
                backward_penalty = count_backward(permutation, self.dag)
                wandb.log({"permutation/backward_penalty": backward_penalty})

                # log the Birkhoff polytope
                if (
                    len(permutation) <= 4
                    and self.birkhoff_config
                    and (
                        (j + epoch * self.permutation_frequency)
                        % self.birkhoff_config.frequency
                        == 0
                    )
                ):
                    batch = next(iter(self.perm_dataloader))
                    img = visualize_birkhoff_polytope(
                        permutation_model=self.permutation_learning_module,
                        num_samples=self.birkhoff_config.num_samples,
                        data=batch,
                        flow_model=self.model,
                        device=self.device,
                    )
                    wandb.log(
                        {
                            "permutation/birkhoff": wandb.Image(
                                img, caption="Birkhoff Polytope"
                            )
                        }
                    )
