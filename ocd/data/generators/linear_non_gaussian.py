from . import SCMGenerator, GraphGenerator
import typing as th
import networkx as nx
import numpy as np


class LinearNonGaussianSCMGenerator(SCMGenerator):
    """
    This is a generator for linear non-gaussian SCMs. It is a subclass of SCMGenerator, and it inherits all of its methods.
    It is assumed that each node is a linear combination of its parents, plus some noise. The noise is assumed to be a subset
    of noise types such as Laplace or Uniform, all of which are non-Gaussian, making the dataset identifiable.
    """

    def __init__(
        self,
        graph_generator: th.Union[GraphGenerator, str],
        graph_generator_args: th.Optional[th.Dict[str, th.Any]] = None,
        seed=None,
        weight: th.Tuple[float, float] = (-1.0, 1.0),
        noise_type: th.Literal["laplace", "uniform"] = "laplace",
    ):
        """
        Args:
            weight (Tuple[float, float], optional): The range of the weights. Defaults to (-1.0, 1.0).
            noise_type (Literal['laplace', 'uniform'], optional): The type of noise. Defaults to 'laplace'.
        """
        super().__init__(graph_generator, graph_generator_args, seed)
        self.weight_low = weight[0]
        self.weight_high = weight[1]
        self.noise_type = noise_type

    def generate_node_functional_parameters(self, dag: nx.DiGraph, node: int, seed: int) -> th.Dict[str, th.Any]:
        # return a dictionary containing one key 'weight' and a number sampled from a uniform distribution between weight_low and weight_high
        # use seed for reproducibility
        np.random.seed(seed)
        return {
            "scale": np.random.uniform(self.weight_low, self.weight_high),
            "bias": np.random.uniform(self.weight_low, self.weight_high),
        }

    def generate_edge_functional_parameters(self, dag: nx.DiGraph, child: int, par: int, seed: int):
        np.random.seed(seed)
        return {"weight": np.random.uniform(self.weight_low, self.weight_high)}

    def generate_noise_functional_parameters(self, dag: nx.DiGraph, node: int, seed: int) -> th.Dict[str, th.Any]:
        return {}

    def get_covariate_from_parents(
        self,
        noise: float,
        parents: th.List[float],
        parent_parameters: th.List[th.Dict[str, th.Any]],
        node_parameters: th.List[th.Dict[str, th.Any]],
    ) -> float:
        return (
            noise * node_parameters["scale"]
            + node_parameters["bias"]
            + sum([p * pp["weight"] for p, pp in zip(parents, parent_parameters)])
        )

    def get_exogenous_noise(self, noise_parameters: th.Dict[str, th.Any], seed: int) -> float:
        np.random.seed(seed)
        if self.noise_type == "laplace":
            return np.random.laplace(0, 1)
        elif self.noise_type == "uniform":
            return np.random.uniform(-1, 1)
        else:
            raise NotImplementedError(f"Noise type {self.noise_type} not implemented")

    def get_covariate_from_parents_signature(
        self,
        node: int,
        parents: th.List[int],
        node_parameters: th.Dict[str, th.Any],
        noise_parameters: th.Dict[str, th.Any],
        parent_parameters: th.List[th.Dict[str, th.Any]],
    ) -> str:
        noise = f"{self.noise_type}({-1 if self.noise_type == 'uniform' else 0}, 1)"
        noise = f"{noise} * {node_parameters['scale']:.2f} + {node_parameters['bias']:.2f}"
        ret = " + ".join([f"x({p}) * {pp['weight']:.2f}" for p, pp in zip(parents, parent_parameters)])
        ret += f" + {noise}"
        return f"x({node}) = {ret}"
