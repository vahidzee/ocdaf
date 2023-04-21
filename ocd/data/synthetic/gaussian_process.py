import typing as th
from ocd.data.scm import SCMGenerator, GraphGenerator
import numpy as np
import dypy as dy
from sklearn.metrics.pairwise import rbf_kernel
import networkx as nx


class GaussianProcessBasedSCMGeberator(SCMGenerator):
    """
    This is a generator based on Gaussian processes.
    For each covariate x_i we have the following formula:

    > x_i = s(Pa(x_i)) * N(0, 1) + t(Pa(x_i))

    Where N(noise_mean, noise_std)      is a standard Gaussian Noise,
    t(x_1, ..., x_k)                    is a non-linear invertible function sampled
                                        from a Gaussian Process with RBF kernel and bandwidth 1
    s(x_1, ..., x_k)                    is a non-linear invertible function sampled from a Gaussian
                                        Process with RBF kernel and bandwidth 1 with a softplus activation
                                        on top to make it positive
    """

    def __init__(
        self,
        graph_generator: th.Union[GraphGenerator, str],
        graph_generator_args: th.Optional[th.Dict[str, th.Any]] = None,
        seed=None,
        noise_std: th.Union[float, th.Tuple[float, float]] = 1.0,
        noise_mean: th.Union[float, th.Tuple[float, float]] = 0.0,
        s_gamma_rbf_kernel: th.Union[float, th.Tuple[float, float]] = 1.0,
        s_variance_rbf_kernel: th.Union[float, th.Tuple[float, float]] = 1.0,
        s_mean_function_weights: th.Union[float, th.Tuple[float, float]] = (0.0, 0.0),
        s_mean_function_activation: th.Optional[th.Union[th.Callable, str]] = None,
        t_gamma_rbf_kernel: th.Union[float, th.Tuple[float, float]] = 1.0,
        t_variance_rbf_kernel: th.Union[float, th.Tuple[float, float]] = 1.0,
        t_mean_function_weights: th.Union[float, th.Tuple[float, float]] = (0.0, 0.0),
        t_mean_function_activation: th.Optional[th.Union[th.Callable, str]] = None,
    ):
        """
        Args:
            weight (Tuple[float, float], optional): The range of the weights. Defaults to (-1.0, 1.0).
            noise_type (Literal['laplace', 'uniform'], optional): The type of noise. Defaults to 'laplace'.
        """
        super().__init__(graph_generator, graph_generator_args, seed)
        self.noise_std = noise_std if not isinstance(noise_std, float) else (noise_std, noise_std)
        self.noise_mean = noise_mean if not isinstance(noise_mean, float) else (noise_mean, noise_mean)

        self.s_gamma_rbf_kernel = (
            s_gamma_rbf_kernel
            if not isinstance(s_gamma_rbf_kernel, float)
            else (s_gamma_rbf_kernel, s_gamma_rbf_kernel)
        )
        self.s_variance_rbf_kernel = (
            s_variance_rbf_kernel
            if not isinstance(s_variance_rbf_kernel, float)
            else (s_variance_rbf_kernel, s_variance_rbf_kernel)
        )
        self.s_mean_function_weights = (
            s_mean_function_weights
            if not isinstance(s_mean_function_weights, float)
            else (s_mean_function_weights, s_mean_function_weights)
        )

        self.t_gamma_rbf_kernel = (
            t_gamma_rbf_kernel
            if not isinstance(t_gamma_rbf_kernel, float)
            else (t_gamma_rbf_kernel, t_gamma_rbf_kernel)
        )
        self.t_variance_rbf_kernel = (
            t_variance_rbf_kernel
            if not isinstance(t_variance_rbf_kernel, float)
            else (t_variance_rbf_kernel, t_variance_rbf_kernel)
        )
        self.t_mean_function_weights = (
            t_mean_function_weights
            if not isinstance(t_mean_function_weights, float)
            else (t_mean_function_weights, t_mean_function_weights)
        )
        # Setup activation functions:
        if s_mean_function_activation is None:
            self.s_mean_function_activation = lambda x: x
        elif isinstance(s_mean_function_activation, str):
            self.s_mean_function_activation = dy.eval(s_mean_function_activation)

        if t_mean_function_activation is None:
            self.t_mean_function_activation = lambda x: x
        elif isinstance(t_mean_function_activation, str):
            self.t_mean_function_activation = dy.eval(t_mean_function_activation)

    def generate_node_functional_parameters(self, dag: nx.DiGraph, node: int, seed: int) -> th.Dict[str, th.Any]:
        np.random.seed(seed)
        s_gamma = np.random.uniform(self.s_gamma_rbf_kernel[0], self.s_gamma_rbf_kernel[1])
        s_var = np.random.uniform(self.s_variance_rbf_kernel[0], self.s_variance_rbf_kernel[1])

        t_gamma = np.random.uniform(self.t_gamma_rbf_kernel[0], self.t_gamma_rbf_kernel[1])
        t_var = np.random.uniform(self.t_variance_rbf_kernel[0], self.t_variance_rbf_kernel[1])

        s_weight = np.random.uniform(self.s_mean_function_weights[0], self.s_mean_function_weights[1])
        t_weight = np.random.uniform(self.t_mean_function_weights[0], self.t_mean_function_weights[1])

        return {
            "t_gamma": t_gamma,
            "t_var": t_var,
            "s_gamma": s_gamma,
            "s_var": s_var,
            "weight_s": s_weight,
            "weight_t": t_weight,
        }

    def generate_edge_functional_parameters(
        self, dag: nx.DiGraph, child: int, par: int, seed: int
    ) -> th.Dict[str, th.Any]:
        np.random.seed(seed)
        return {
            "weight_s": np.random.uniform(self.s_mean_function_weights[0], self.s_mean_function_weights[1]),
            "weight_t": np.random.uniform(self.t_mean_function_weights[0], self.t_mean_function_weights[1]),
        }

    def generate_noise_functional_parameters(self, dag: nx.DiGraph, node: int, seed: int) -> th.Dict[str, th.Any]:
        np.random.seed(seed)
        avg = np.random.uniform(self.noise_mean[0], self.noise_mean[1])
        std = np.random.uniform(self.noise_std[0], self.noise_std[1])
        return {"mean": avg, "std": std}

    def get_covariate_from_parents(
        self,
        noise: np.array,
        parents: th.List[np.array],
        parent_parameters: th.List[th.Dict[str, th.Any]],
        node_parameters: th.List[th.Dict[str, th.Any]],
    ) -> np.array:
        s_mean = sum([node_parameters["weight_s"]] + [p * pp["weight_s"] for p, pp in zip(parents, parent_parameters)])
        s_mean = self.s_mean_function_activation(s_mean)
        t_mean = sum([node_parameters["weight_t"]] + [p * pp["weight_t"] for p, pp in zip(parents, parent_parameters)])
        t_mean = self.t_mean_function_activation(t_mean)

        if len(parents) > 0:
            parents_np = np.stack(parents).T
            s_kernel = node_parameters["s_var"] * rbf_kernel(parents_np, gamma=node_parameters["s_gamma"])
            t_kernel = node_parameters["s_var"] * rbf_kernel(parents_np, gamma=node_parameters["s_gamma"])

            # Sample x_values from a multivariate normal distribution with mean x_mean and covariance x_kernel
            s_values = np.random.multivariate_normal(s_mean, s_kernel)
            t_values = np.random.multivariate_normal(t_mean, t_kernel)
        else:
            s_values = s_mean * np.ones(noise.shape)
            t_values = t_mean * np.ones(noise.shape)

        # Turn all the s_values into positive by passing it through a softplus function
        s_values = np.log(1 + np.exp(s_values))

        return t_values + s_values * noise

    def get_exogenous_noise(self, noise_parameters: th.Dict[str, th.Any], seed: int, n_samples: int = 1) -> np.array:
        np.random.seed(seed)
        return np.random.normal(noise_parameters["mean"], noise_parameters["std"], n_samples)

    def get_covariate_from_parents_signature(
        self,
        node: int,
        parents: th.List[int],
        node_parameters: th.Dict[str, th.Any],
        noise_parameters: th.Dict[str, th.Any],
        parent_parameters: th.List[th.Dict[str, th.Any]],
    ) -> str:
        noise = f"N({noise_parameters['mean']:.2f}, {noise_parameters['std']:.2f})"
        s_mean = "+".join(
            [f"{node_parameters['weight_s']:.2f}"]
            + [f"x({p}) * {pp['weight_s']:.2f}" for p, pp in zip(parents, parent_parameters)]
        )
        t_mean = "+".join(
            [f"{node_parameters['weight_t']:.2f}"]
            + [f"x({p}) * {pp['weight_t']:.2f}" for p, pp in zip(parents, parent_parameters)]
        )
        s_kernel = f"RBF(gamma={node_parameters['s_gamma']:.2f}, bandwidth={node_parameters['s_var']:.2f})"
        t_kernel = f"RBF(gamma={node_parameters['t_gamma']:.2f}, bandwidth={node_parameters['t_var']:.2f})"
        return f"GP(mean=[{t_mean}], var={t_kernel}) + GP(mean=[{s_mean}], var={s_kernel}) * {noise}"
