from typing import Optional, Callable, Iterable, Union, List, Literal
from omegaconf import MISSING
from pydantic import BaseModel, validator, Field
import 
import torch



class WandBConfig(BaseModel):
    pass


class CheckpointingConfig(BaseModel):
    frequency: int
    directory: str


class BirkhoffConfig(BaseModel):
    pass


class DataVisualizer(BaseModel):
    pass


class SchedulerConfig(BaseModel):
    flow_frequency: int
    permutation_frequency: int
    flow_lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]]
    permutation_lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]]



class SoftSinkhornConfig(BaseModel):
    method: str = "soft-sinkhorn"
    temp: float 
    iters: int

class GumbelTopKConfig(BaseModel):
    method: str = "gumbel-top-k"
    num_samples: int

class GumbelSinkhornConfig(BaseModel):
    method: str = "gumbel-sinkhorn"
    temp: float
    iters: int
    num_samples: int

class TrainingConfig(BaseModel):
    cuda: bool
    checkpointing: Optional[CheckpointingConfig]

    # training loop configurations
    max_epochs: int
    flow_optimizer: Callable[[Iterable], torch.optim.Optimizer]
    permutation_optimizer: Callable[[Iterable], torch.optim.Optimizer]

    scheduler: SchedulerConfig
    permutation: Union[GumbelSinkhornConfig, GumbelTopKConfig, SoftSinkhornConfig]

    # tracking configurations
    data_visualizer: Optional[DataVisualizer]
    brikhoff: Optional[BirkhoffConfig]


class ModelConfig(BaseModel):
    in_features: int
    layers: List[int]
    num_transforms: int = 1
    residual: bool = False  # TODO: test and drop if not used
    activation: torch.nn.Module = torch.nn.LeakyReLU()
    # additional flow args
    additive: bool = False
    normalization: Optional[torch.nn.Module]  # TODO: test the actnorm / tanh
    # base distribution
    base_distribution: torch.distributions.Distribution = torch.distributions.Normal()
    # ordering
    ordering: Optional[torch.IntTensor]
    reversed_ordering: bool = False
    # gamma config
    

class MainConfig(BaseModel):
    training: TrainingConfig
    
    wandb: Optional[WandBConfig]
