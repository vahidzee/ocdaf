from typing import Optional, Callable, Iterable, Union, List, Literal, Tuple
from omegaconf import MISSING
from pydantic import BaseModel, validator
import torch
import numpy as np
from functools import partial
from ocd.data.base_dataset import OCDDataset

class RandomGenerator: # TODO move it to utils + documentations
    def __init__(self, noise_type: str, seed: Optional[int], *args, **kwargs):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        if not hasattr(rng, noise_type):
            raise ValueError(f"Unknown noise type {noise_type}")
        
        self.rng = partial(getattr(rng, noise_type), *args, **kwargs)
    
    def __call__(self, size: Union[int, Tuple[int, ...]]):
        return self.rng(size=size)


class RealworldConfig(BaseModel):
    name: Literal["adni", "sachs"]


class SemiSyntheticConfig(BaseModel):
    name: str = "syntren"
    data_id: int

    @validator("data_id", always=True)
    def validate_data_id(cls, value, values):
        if values['name'] == "syntren":
            assert 0 <= value <= 9, "data_id must be between 0 and 9"
        return value


class GraphConfig(BaseModel):
    graph_type:  Literal["full", "erdos", "chain"]
    num_nodes: int
    seed: Optional[int]

class ParametricSyntheticConfig(BaseModel):
    name: Optional[str]
    num_samples: int
    graph: GraphConfig
    noise: RandomGenerator
    link: Literal["sinusoid", "cubic", "linear"]
    link_generator: RandomGenerator

class NonParametricSyntheticConfig(BaseModel):
    name: Optional[str]
    num_samples: int
    graph: GraphConfig
    seed: Optional[int]


class DataConfig(BaseModel):
    dataset: Union[RealworldConfig, SemiSyntheticConfig, NonParametricSyntheticConfig, ParametricSyntheticConfig, OCDDataset]
    batch_size: int
    standard: bool
    reject_outliers: bool



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
