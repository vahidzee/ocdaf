"""
This file contains a fast parametric SCM generator
that can generte datasets with parametric assumptions.
"""

from ocd.data.scm import SCMGenerator, GraphGenerator
import typing as th
import networkx as nx
import numpy
import numpy as np
import dypy as dy
import math

class ParametricSCMGenerator(SCMGenerator):
    """
    This is an SCM generator that generates SCMs with the following function formulas:

    x_i = t(Pa_i) + s(Pa_i) * Noise(**Parameters)

    where t is a non-linear invertible function and
    s is a positive function, and Noise is a noise function generated
    from a noise type and its parameters that can be specified in the constructor.

    weight_s and weight_t are sampled from Uniform(weight_s[0], weight_s[1]) and
    Uniform(weight_t[0], weight_t[1]) respectively.
    The following functions are used for t and s:

    s_inp(x_1, x_2, ..., x_k) = s_1 * x_1 + s_2 * x_2 + ... + s_k * x_k + s_0
    t_inp(x_1, x_2, ..., x_k) = t_1 * x_1 + t_2 * x_2 + ... + t_k * x_k + t_0

    where s_i and t_i are sampled from Uniform(weight_s[0], weight_s[1]) and
    Uniform(weight_t[0], weight_t[1]) respectively.

    Finally, using the s_function and t_function, we get the actual s and t functions:

    s = s_function(s_inp), t = t_function(t_inp)

    For printing out the formulas, we use the s_function_signature and t_function_signature
    """

    def __init__(
        self,
        graph_generator: th.Union[GraphGenerator, str],
        graph_generator_args: th.Optional[th.Dict[str, th.Any]] = None,
        seed=None,
        
        # The noise functions
        noise_type: th.Literal["laplace", "uniform", "normal"] = "laplace",
        noise_parameters: th.Optional[th.Dict[str, th.Any]] = None,
        
        weight_s: th.Union[float, th.Tuple[float, float]] = (-1.0, 1.0),
        weight_t: th.Union[float, th.Tuple[float, float]] = (-1.0, 1.0),
        s_function: th.Optional[th.Union[th.Callable, str, th.Dict[str, str]]] = None,
        t_function: th.Optional[th.Union[th.Callable, str, th.Dict[str, str]]] = None,
        s_function_signature: th.Optional[str] = None,
        t_function_signature: th.Optional[str] = None,
    ):
        """
        Create a premade SCM generator for simulated data.

        Args:
            graph_generator (GraphGenerator): A standard graph generator for the structure of SCM.
            seed (int, optional): A seed for reproducibility. Defaults to None.
            std (th.Tuple[float, float], optional): The range of the noise standard deviation -- Defaults to (0.1, 1.0).
            mean (th.Tuple[float, float], optional): The range of the noise mean -- Defaults to (-1, 1.0).
            weight_t (th.Tuple[float, float], optional): The range of the parameters in s -- Defaults to (-1.0, 1.0).
            weight_s (th.Tuple[float, float], optional): The range of the parameters in t -- Defaults to (-1.0, 1.0).
        """
        super().__init__(graph_generator, graph_generator_args, seed)
        
        # Handle the weights
        if isinstance(weight_t, float):
            weight_t = (weight_t, weight_t)
        if isinstance(weight_s, float):
            weight_s = (weight_s, weight_s)
        self.weight_t = weight_t
        self.weight_s = weight_s
        
        # Handle the noise parameters
        for key, item in noise_parameters.items():
            if isinstance(item, float):
                noise_parameters[key] = (item, item)
        self.noise_type = noise_type
        self.noise_parameters = noise_parameters          

        # Add useful contexts
        dy.register_context(numpy)
        dy.register_context(math)

        # Setup s function
        if s_function is None:
            self.s_function = lambda x: numpy.log(1 + numpy.exp(x))
            self.s_function_signature = lambda x: f"softplus({x})"
        else:
            if isinstance(s_function, str):
                self.s_function = dy.eval_function(s_function)
            elif isinstance(s_function, dict):
                self.s_function = dy.eval_function(**s_function)
            else:
                self.s_function = s_function
            if s_function_signature is None:
                raise ValueError("s_function_signature must be provided if s_function is provided.")
            self.s_function_signature = lambda x: f"{s_function_signature}({x})"

        # Setup t function
        if t_function is None:
            self.t_function = lambda x: numpy.log(1 + numpy.exp(x))
            self.t_function_signature = lambda x: f"softplus({x})"
        else:
            if isinstance(t_function, str):
                self.t_function = dy.eval_function(t_function)
            elif isinstance(t_function, dict):
                self.t_function = dy.eval_function(**t_function)
            else:
                self.t_function = t_function

            if t_function_signature is None:
                raise ValueError("t_function_signature must be provided if t_function is provided.")
            self.t_function_signature = lambda x: f"{t_function_signature}({x})"

    def generate_edge_functional_parameters(self, dag: nx.DiGraph, child: int, par: int, seed: int):
        numpy.random.seed(seed)
        return dict(
            weight_t=numpy.random.uniform(self.weight_t[0], self.weight_t[1]),
            weight_s=numpy.random.uniform(self.weight_s[0], self.weight_s[1]),
        )

    def generate_node_functional_parameters(self, dag: nx.DiGraph, node: int, seed: int) -> th.Dict[str, th.Any]:
        numpy.random.seed(seed)
        return dict(
            weight_t=numpy.random.uniform(self.weight_t[0], self.weight_t[1]),
            weight_s=numpy.random.uniform(self.weight_s[0], self.weight_s[1]),
        )

    def generate_noise_functional_parameters(self, dag: nx.DiGraph, node: int, seed: int) -> th.Dict[str, th.Any]:
        numpy.random.seed(seed)
        ret = {}
        for key, item in self.noise_parameters.items():
            ret[key] = numpy.random.uniform(item[0], item[1])  
        return ret

    def get_exogenous_noise(
        self, noise_parameters: th.Dict[str, th.Any], seed: int, n_samples: int = 1
    ) -> numpy.array:
        numpy.random.seed(seed)
        return getattr(numpy.random, self.noise_type)(**noise_parameters, size=n_samples)

    def get_covariate_from_parents(
        self,
        noise: np.array,
        parents: th.List[float],
        parent_parameters: th.List[th.Dict[str, th.Any]],
        node_parameters: th.List[th.Dict[str, th.Any]],
    ) -> numpy.array:
        t_input = sum([p * pp["weight_t"] for p, pp in zip(parents, parent_parameters)]) + node_parameters["weight_t"]

        if isinstance(t_input, float):
            t_input = t_input * np.ones_like(noise)

        t = self.t_function(t_input)

        s_input = sum([p * pp["weight_s"] for p, pp in zip(parents, parent_parameters)]) + node_parameters["weight_s"]
        if isinstance(s_input, float):
            s_input = s_input * np.ones_like(noise)
        s = self.s_function(s_input)
        return t + s * noise

    def get_covariate_from_parents_signature(
        self,
        node: int,
        parents: th.List[int],
        node_parameters: th.Dict[str, th.Any],
        noise_parameters: th.Dict[str, th.Any],
        parent_parameters: th.List[th.Dict[str, th.Any]],
    ) -> str:
        sum_list = "+".join([f'x({p})*{pp["weight_t"]:.2f}' for p, pp in zip(parents, parent_parameters)])
        t = self.t_function_signature(f"{sum_list}+{node_parameters['weight_t']:.2f}")

        sum_list = "+".join([f'x({p})*{pp["weight_s"]:.2f}' for p, pp in zip(parents, parent_parameters)])
        s = self.s_function_signature(f"{sum_list}+{node_parameters['weight_s']:.2f}")
        noise_repr = ",".join([f"{key}={val:.2f}" for key, val in noise_parameters.items()])
        noise_repr = f"{self.noise_type}({noise_repr})"
        return f"x({node}) = {t} + {s} * {noise_repr}"
