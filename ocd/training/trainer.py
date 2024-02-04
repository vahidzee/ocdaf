import torch

from ocd.config import (
    GumbelSinkhornConfig,
    GumbelTopKConfig,
    SoftSinkhornConfig,
    ContrastiveDivergenceConfig,
    DataVisualizer,
    BirkhoffConfig,
)

from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from typing import Union, Callable, Iterable, Optional
from ocd.models.oslow import OSlow
from ocd.training.permutation import GumbelTopK, ContrastiveDivergence
from ocd.visualization.birkhoff import visualize_birkhoff_polytope


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
        flow_lr_scheduler: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ],
        permutation_lr_scheduler: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ],
        permutation_learning_config: Union[
            GumbelSinkhornConfig, GumbelTopKConfig, SoftSinkhornConfig, ContrastiveDivergenceConfig
        ],
        data_visualizer_config: Optional[DataVisualizer] = None,
        birkhoff_config: Optional[BirkhoffConfig] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.max_epochs = max_epochs
        self.model = model.to(device)
        permutation_learning_kwargs = permutation_learning_config.model_dump()
        permutation_learning_kwargs.pop("method")
        if isinstance(permutation_learning_config, ContrastiveDivergenceConfig):
            self.permutation_learning_module = ContrastiveDivergence(
                model.in_features, **permutation_learning_kwargs
            ).to(device)
        elif isinstance(permutation_learning_config, GumbelTopKConfig):
            self.permutation_learning_module = GumbelTopK(
                model.in_features, **permutation_learning_kwargs
            ).to(device)
        else:
            # TODO: update and add other baselines for ablation study
            raise ValueError("permutation_learning_config must be of type GumbelSinkhornConfig or ContrastiveDivergenceConfig")
        
        self.dataloader = dataloader
        self.permutation_learning_config = permutation_learning_config
        self.data_visualizer_config = data_visualizer_config
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

        # TODO create checkpointing
        self.checkpointing = None
        # tracking configurations
        self.data_visualizer = None
        self.brikhoff = None

    def flow_train_step(self):
        self.permutation_learning_module._gamma.requires_grad = False
        for (batch,) in self.dataloader:
            batch = batch.to(self.model.device)
            # permutations = self.permutation_learning_module.sample_hard_permutations(
            #     batch.shape[0]
            # )
            # permutations = torch.unique(permutations, dim=0)
            # # sample from permutations with replacement with the batch size
            # # shape: (batch_size, d, d)
            # permutations = permutations[
            #     torch.randint(
            #         0,
            #         permutations.shape[0],
            #         (batch.shape[0],),
            #         device=permutations.device,
            #     )
            # ]
            # TODO test both losses
            self.flow_optimizer.zero_grad()
            # loss = -(self.model.log_prob(batch, permutations)).mean()
            loss = self.permutation_learning_module.flow_learning_loss(self.model, batch)
            loss.backward()
            self.flow_optimizer.step()
        self.permutation_learning_module._gamma.requires_grad = True
        return loss

    def permutation_train_step(self):
        # stop gradient model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for (batch,) in self.dataloader:
            batch = batch.to(self.model.device)
            self.permutation_optimizer.zero_grad()
            loss = self.permutation_learning_module.permutation_learning_loss(self.model, batch)
            loss.backward()
            self.permutation_optimizer.step()
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

        flow_progress_bar = tqdm(
            total=true_epochs * self.flow_frequency,
            desc="training the flow",
            position=0,
        )
        permutation_progress_bar = tqdm(
            total=true_epochs * self.permutation_frequency,
            desc="training the permutation",
            position=1,
        )
        histories = {"flow_loss": [], "permutation_loss": [], "birkhoffs": []}
        for epoch in range(true_epochs):
            for i in range(self.flow_frequency):
                loss = self.flow_train_step()
                self.flow_scheduler.step()
                flow_progress_bar.update(1)
                flow_progress_bar.set_postfix({"flow loss": loss.item()})
                histories["flow_loss"].append(loss.item())

            for j in range(self.permutation_frequency):
                loss = self.permutation_train_step()
                self.permutation_scheduler.step()
                permutation_progress_bar.update(1)
                permutation_progress_bar.set_postfix({"permutation loss": loss.item()})
                histories["permutation_loss"].append(loss.item())
                if self.birkhoff_config and (
                    (j + epoch * self.permutation_frequency)
                    % self.birkhoff_config.frequency
                    == 0
                ):
                    batch = next(iter(self.dataloader))[0]
                    img = visualize_birkhoff_polytope(
                        permutation_model=self.permutation_learning_module,
                        num_samples=self.birkhoff_config.num_samples,
                        data=batch,
                        flow_model=self.model,
                        device=self.device,
                    )
                    plt.imshow(img)
                    plt.show()
                    histories["birkhoffs"].append(img)
        return histories
