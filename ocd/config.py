"""
This file contains the entire configuration of the project.
Everything lies within the MainConfig class, which is a Pydantic model.
"""
from typing import Optional, Callable, Iterable, Union, List, Literal, Tuple
from pydantic import model_validator, field_validator
from pydantic import BaseModel as PydanticBaseModel
import torch
import numpy as np
from functools import partial
from ocd.data.base_dataset import OCDDataset
from ocd.data.synthetic.utils import RandomGenerator
import networkx as nx


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class RealworldConfig(BaseModel):
    name: Literal["adni", "sachs"]


class SemiSyntheticConfig(BaseModel):
    name: str = "syntren"
    data_id: int

    @field_validator("data_id")
    def validate_data_id(cls, value, values):
        if values["name"] == "syntren":
            assert 0 <= value <= 9, "data_id must be between 0 and 9"
        return value


class GraphConfig(BaseModel):
    graph_type: Literal["full", "erdos", "chain"]
    num_nodes: int
    seed: Optional[int] = None
    enforce_ordering: Optional[list] = None


class ParametricSyntheticConfig(BaseModel):
    num_samples: int
    graph: GraphConfig
    noise_generator: RandomGenerator
    link_generator: RandomGenerator
    link: Literal["sinusoid", "cubic", "linear"] = ("sinusoid",)
    perform_normalization: bool = True
    additive: bool = False
    post_non_linear_transform: Optional[
        Literal["exp", "softplus", "x_plus_sin", "sinusoid", "nonparametric"]
    ] = None


class NonParametricSyntheticConfig(BaseModel):
    name: Optional[str] = None
    num_samples: int
    graph: GraphConfig
    seed: Optional[int] = None
    post_non_linear_transform: Optional[
        Literal["exp", "softplus", "x_plus_sin", "sinusoid", "nonparametric"]
    ] = None
    noise_generator: RandomGenerator
    s_rbf_kernel_gamma: float = 1.0
    t_rbf_kernel_gamma: float = 1.0
    invertibility_coefficient: float = 0.0
    perform_normalization: bool = True
    additive: bool = False


class DataConfig(BaseModel):
    dataset: Union[
        RealworldConfig,
        SemiSyntheticConfig,
        ParametricSyntheticConfig,
        NonParametricSyntheticConfig,
    ]
    standard: bool
    reject_outliers: bool
    outlier_threshold: float = 3.0


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
    frequency: int
    num_samples: int


class DataVisualizer(BaseModel):
    pass


class SchedulerConfig(BaseModel):
    flow_frequency: int
    permutation_frequency: int
    flow_lr_scheduler: Optional[
        Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]
    ] = None
    permutation_lr_scheduler: Optional[
        Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]
    ] = None


class SoftSortConfig(BaseModel):
    method: str = "soft-sort"
    temp: float
    parameterization_type: Literal["vanilla", "sigmoid"]
    uniform: bool = False


class ContrastiveDivergenceConfig(BaseModel):
    method: str = "contrastive-divergence"
    num_samples: int
    parameterization_type: Literal["vanilla", "sigmoid"]
    uniform: bool = False
    chunk_size: Optional[int] = None


class SoftSinkhornConfig(BaseModel):
    method: str = "soft-sinkhorn"
    temp: float
    iters: int
    parameterization_type: Literal["vanilla", "sigmoid"]
    uniform: bool = False


class GumbelTopKConfig(BaseModel):
    method: str = "gumbel-top-k"
    parameterization_type: Literal["vanilla", "sigmoid"]
    num_samples: int
    chunk_size: Optional[int] = None
    different_flow_loss: bool = False
    uniform: bool = False


class GumbelSinkhornStraightThroughConfig(BaseModel):
    method: str = "straight-through-sinkhorn"
    temp: float
    iters: int
    parameterization_type: Literal["vanilla", "sigmoid"]
    uniform: bool = False


class TrainingConfig(BaseModel):
    device: str
    checkpointing: Optional[CheckpointingConfig] = None

    # training loop configurations
    max_epochs: int
    flow_batch_size: int
    permutation_batch_size: int
    flow_optimizer: Callable[[Iterable], torch.optim.Optimizer]
    permutation_optimizer: Callable[[Iterable], torch.optim.Optimizer]

    scheduler: SchedulerConfig
    permutation: Union[GumbelSinkhornStraightThroughConfig, GumbelTopKConfig, SoftSinkhornConfig]
    gumbel_std: Optional[float] = 1.
    brikhoff: Optional[BirkhoffConfig] = None

    @field_validator("device")
    def validate_device(cls, value):
        values = value.split(":")
        assert values[0] in ["cpu", "cuda"], "device must be either cpu or cuda"
        if len(values) > 1 and values[1] != "":
            assert int(values[1]) >= 0, "device number must be positive"
        return value


class ModelConfig(BaseModel):
    in_features: int
    layers: List[int]
    dropout: Optional[float]
    residual: bool = False
    activation: torch.nn.Module = torch.nn.LeakyReLU()
    additive: bool = False
    num_transforms: int
    normalization: Optional[Callable[[int], torch.nn.Module]]
    base_distribution: torch.distributions.Distribution = torch.distributions.Normal(
        0.0, 1.0
    )
    ordering: Optional[torch.IntTensor] = None

    ###### Post Non Linear Transform ######
    num_post_nonlinear_transforms: int = 0  # change if you want to consider PNL models
    num_bins: int = 10
    tail_bound: float = 10.0
    identity_init: bool = False
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3
    normalization: Optional[Callable[[int], torch.nn.Module]] = None


class MainConfig(BaseModel):
    trainer: TrainingConfig
    data: Union[DataConfig, OCDDataset]
    model: ModelConfig
    wandb: WandBConfig
    out_dir: str
    test_run: bool

    @model_validator(mode="after")
    def validate_model(self):
        name_features = {"adni": 8, "sachs": 11, "syntren": 20}
        data = self.data
        if isinstance(data.dataset, ParametricSyntheticConfig) or isinstance(
            data.dataset, NonParametricSyntheticConfig
        ):
            if self.model.in_features is None:
                self.model.in_features = data.dataset.graph.num_nodes
            assert (
                self.model.in_features == data.dataset.graph.num_nodes
            ), "in_features must be equal to the number of nodes in the graph"

        elif isinstance(data.dataset, RealworldConfig) or isinstance(
            data.dataset, SemiSyntheticConfig
        ):
            if self.model.in_features is None:
                self.model.in_features = name_features[data.dataset.name]
            assert (
                self.model.in_features == name_features[data.dataset.name]
            ), "in_features must be equal to the number of nodes in the graph"
        elif isinstance(data.dataset, OCDDataset):
            if self.model.in_features is None:
                self.model.in_features = len(data.dataset.dag.nodes)
            assert self.model.in_features == len(
                data.dataset.dag.nodes
            ), "in_features must be equal to the number of nodes in the graph"

        if self.model.ordering is not None:
            assert set(self.model.ordering) == set(
                range(self.model.in_features)
            ), "ordering must be a permutation of range(in_features)"

        return self
