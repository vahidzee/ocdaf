from .base_scm_generator import SCMGenerator
from .graph_generator import GraphGenerator
import typing as th
import networkx as nx
import numpy as np


class InvertibleModulatedGaussianSCMGenerator(SCMGenerator):
    """
    This is an SCM generator that generates SCMs with the following function formulas:

    x_i = f(Pa_i) + g(Pa_i) * N(mean, std)

    where f and g are non-linear invertible functions of the following form:

    func(x_1, x_2, ..., x_k) = exp(sigmoid(w_1 * x_1 + w_2 * x_2 + ... + w_k * x_k + b))

    All the w_i values for f are sampled from Uniform(weight_f[0], weight_f[1])
    and all the w_i values for g are sampled from Uniform(weight_g[0], weight_g[1]).

    Mean and std of the noise are sampled from Uniform(mean[0], mean[1]) and Uniform(std[0], std[1]) respectively.
    """

    def __init__(
        self,
        graph_generator: th.Union[GraphGenerator, str],
        graph_generator_args: th.Optional[th.Dict[str, th.Any]] = None,
        seed=None,
        std: th.Union[float, th.Tuple[float, float]] = 1,
        mean: th.Union[float, th.Tuple[float, float]] = 0,
        weight_f: th.Union[float, th.Tuple[float, float]] = (-1.0, 1.0),
        weight_g: th.Union[float, th.Tuple[float, float]] = (-1.0, 1.0),
    ):
        """
        Create a premade SCM generator for simulated data.

        Args:
            graph_generator (GraphGenerator): A standard graph generator for the structure of SCM.
            seed (int, optional): A seed for reproducibility. Defaults to None.
            std (th.Tuple[float, float], optional): The range of the noise standard deviation -- Defaults to (0.1, 1.0).
            mean (th.Tuple[float, float], optional): The range of the noise mean -- Defaults to (-1, 1.0).
            weight_f (th.Tuple[float, float], optional): The range of the parameters in f -- Defaults to (-1.0, 1.0).
            weight_g (th.Tuple[float, float], optional): The range of the parameters in g -- Defaults to (-1.0, 1.0).
        """
        super().__init__(graph_generator, graph_generator_args, seed)
        # check if weight_f and weight_g are float values
        if isinstance(weight_f, float):
            weight_f = (weight_f, weight_f)
        if isinstance(weight_g, float):
            weight_g = (weight_g, weight_g)
        if isinstance(std, float):
            std = (std, std)
        if isinstance(mean, float):
            mean = (mean, mean)
        self.weight_f = weight_f
        self.weight_g = weight_g
        self.std = std
        self.mean = mean

    def generate_edge_functional_parameters(self, dag: nx.DiGraph, child: int, par: int, seed: int):
        np.random.seed(seed)
        return dict(
            weight_f=np.random.uniform(self.weight_f[0], self.weight_f[1]),
            weight_g=np.random.uniform(self.weight_g[0], self.weight_g[1]),
        )

    def generate_node_functional_parameters(self, dag: nx.DiGraph, node: int, seed: int) -> th.Dict[str, th.Any]:
        np.random.seed(seed)
        return dict(
            weight_f=np.random.uniform(self.weight_f[0], self.weight_f[1]),
            weight_g=np.random.uniform(self.weight_g[0], self.weight_g[1]),
        )

    def generate_noise_functional_parameters(self, dag: nx.DiGraph, node: int, seed: int) -> th.Dict[str, th.Any]:
        np.random.seed(seed)
        return dict(
            std=np.random.uniform(self.std[0], self.std[1]),
            mean=np.random.uniform(self.mean[0], self.mean[1]),
        )

    def get_exogenous_noise(self, noise_parameters: th.Dict[str, th.Any], seed: int) -> float:
        np.random.seed(seed)
        return np.random.normal(noise_parameters["mean"], noise_parameters["std"])

    def sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def get_covariate_from_parents(
        self,
        noise: float,
        parents: th.List[float],
        parent_parameters: th.List[th.Dict[str, th.Any]],
        node_parameters: th.List[th.Dict[str, th.Any]],
    ) -> float:
        f = np.exp(
            self.sigmoid(
                sum([p * pp["weight_f"] for p, pp in zip(parents, parent_parameters)]) + node_parameters["weight_f"]
            )
        )
        g = np.exp(
            self.sigmoid(
                sum([p * pp["weight_g"] for p, pp in zip(parents, parent_parameters)]) + node_parameters["weight_g"]
            )
        )
        return f + g * noise

    def get_covariate_from_parents_signature(
        self,
        node: int,
        parents: th.List[int],
        node_parameters: th.Dict[str, th.Any],
        noise_parameters: th.Dict[str, th.Any],
        parent_parameters: th.List[th.Dict[str, th.Any]],
    ) -> str:
        sum_list = "+".join([f'x({p})*{pp["weight_f"]:.2f}' for p, pp in zip(parents, parent_parameters)])
        f = f"exp(sigmoid({sum_list}+{node_parameters['weight_f']:.2f}))"

        sum_list = "+".join([f'x({p})*{pp["weight_g"]:.2f}' for p, pp in zip(parents, parent_parameters)])
        g = f"exp(sigmoid({sum_list}+{node_parameters['weight_g']:.2f}))"

        return f"x({node}) = {f} + {g} * N({noise_parameters['mean']:.2f},{noise_parameters['std']:.2f})"
