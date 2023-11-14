from typing import Optional, Callable, Iterable, Union, List, Literal, Tuple
from pydantic import model_validator, field_validator
from pydantic import BaseModel as PydanticBaseModel
import torch
import numpy as np
from functools import partial
from ocd.data.base_dataset import OCDDataset

class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

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

    @field_validator("data_id")
    def validate_data_id(cls, value, values):
        if values['name'] == "syntren":
            assert 0 <= value <= 9, "data_id must be between 0 and 9"
        return value


class GraphConfig(BaseModel):
    graph_type:  Literal["full", "erdos", "chain"]
    num_nodes: int
    seed: Optional[int] = None

class ParametricSyntheticConfig(BaseModel):
    name: Optional[str] = None
    num_samples: int
    graph: GraphConfig
    noise: RandomGenerator
    link: Literal["sinusoid", "cubic", "linear"]
    link_generator: RandomGenerator


class NonParametricSyntheticConfig(BaseModel):
    name: Optional[str] = None
    num_samples: int
    graph: GraphConfig
    seed: Optional[int] = None


class DataConfig(BaseModel):
    dataset: Union[RealworldConfig, SemiSyntheticConfig, NonParametricSyntheticConfig, ParametricSyntheticConfig, OCDDataset]
    batch_size: int
    standard: bool
    reject_outliers: bool



class WandBConfig(BaseModel):
    project: str
    entity: Optional[str] = None
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    resume: bool = False


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
    flow_lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]] = None
    permutation_lr_scheduler: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]] = None



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
    checkpointing: Optional[CheckpointingConfig] = None

    # training loop configurations
    max_epochs: int
    flow_optimizer: Callable[[Iterable], torch.optim.Optimizer]
    permutation_optimizer: Callable[[Iterable], torch.optim.Optimizer]

    scheduler: SchedulerConfig
    permutation: Union[GumbelSinkhornConfig, GumbelTopKConfig, SoftSinkhornConfig]

    # tracking configurations
    data_visualizer: Optional[DataVisualizer] = None
    brikhoff: Optional[BirkhoffConfig] = None

class ModelConfig(BaseModel):
    in_features: Optional[int] = None
    layers: List[int]
    num_transforms: int = 1
    residual: bool = False  # TODO: test and drop if not used
    activation: torch.nn.Module = torch.nn.LeakyReLU()
    # additional flow args
    additive: bool = False
    normalization: Optional[torch.nn.Module] = None # TODO: test the actnorm / tanh
    # base distribution
    base_distribution: torch.distributions.Distribution = torch.distributions.Normal(0., 1.)
    # ordering
    ordering: Optional[torch.IntTensor] = None
    reversed_ordering: bool = False
    # gamma config
    

class MainConfig(BaseModel):
    trainer: TrainingConfig
    data: DataConfig
    model: ModelConfig
    wandb: WandBConfig
    out_dir: str
    test_run: bool

    @model_validator(mode='after')
    def validate_model(self):
        name_features = {'adni': 8, 'sachs': 11, 'syntren': 20}
        data = self.data
        if isinstance(data.dataset, ParametricSyntheticConfig) or isinstance(data.dataset, NonParametricSyntheticConfig):
            if self.model.in_features is None:
                self.model.in_features = data.dataset.graph.num_nodes
            assert self.model.in_features == data.dataset.graph.num_nodes, "in_features must be equal to the number of nodes in the graph"

        elif isinstance(data.dataset, RealworldConfig) or isinstance(data.dataset, SemiSyntheticConfig):
            if self.model.in_features is None:
                self.model.in_features = name_features[data.dataset.name]
            assert self.model.in_features == name_features[data.dataset.name], "in_features must be equal to the number of nodes in the graph"
        elif isinstance(data.dataset, OCDDataset):
            if self.model.in_features is None:
                self.model.in_features = len(data.dataset.dag.nodes)
            assert self.model.in_features == len(data.dataset.dag.nodes), "in_features must be equal to the number of nodes in the graph"
        
        if self.model.ordering is not None:
            assert set(self.model.ordering) == set(range(self.model.in_features)), "ordering must be a permutation of range(in_features)"

        return self




