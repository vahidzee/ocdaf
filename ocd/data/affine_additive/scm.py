from ocd.data.scm import SCM, SCMGenerator
import typing as th
import random
import numpy as np


SCM_FUNCTION_TYPES = th.Literal["linear_additive", None]


class AffineAdditiveSCMGenerator(SCMGenerator):
    def __init__(
        self,
        function_type: SCM_FUNCTION_TYPES = None,
        noise_type: th.Literal["gaussian", "laplace", None] = None,
        **parent_kwargs,
    ):
        def get_noise_parameters(dag, v, seed, **kwargs):
            np.random.seed(seed)
            return dict(
                noise_mean=np.random.uniform(kwargs["noise_mean_low"], kwargs["noise_mean_high"]),
                noise_std=np.random.uniform(kwargs["noise_std_low"], kwargs["noise_std_high"]),
            )

        if function_type == "linear_additive":

            def get_weight_node(dag, v, seed, **kwargs):
                np.random.seed(seed)
                return dict(weight=np.random.uniform(kwargs["weight_low"], kwargs["weight_high"]))

            def get_weight_edge(dag, v, par, seed, **kwargs):
                np.random.seed(seed)
                return dict(weight=np.random.uniform(kwargs["weight_low"], kwargs["weight_high"]))

            def get_covariate_from_parents(inputs, params):
                return sum(t[0] * (1 if i == 0 else t[1]["weight"]) for i, t in enumerate(list(zip(inputs, params))))

            def get_covariate_from_parents_signature(inputs, params):
                sep = " + "
                ret = f'{noise_type}({inputs[0]["noise_mean"]:.2f}, {inputs[0]["noise_std"]:.2f})'
                addition = sep.join(f'{w["weight"]:.2f}*x({i})' for i, w in zip(inputs[1:], params[1:]))
                if addition:
                    ret += sep + addition
                return ret

        else:
            raise NotImplementedError("function_type {} not in {}".format(function_type, SCM_FUNCTION_TYPES.__args__))

        if noise_type == "gaussian":

            def get_noise(seed, **kwargs):
                np.random.seed(seed)
                return np.random.normal(kwargs["noise_mean"], kwargs["noise_std"])

        elif noise_type == "laplace":

            def get_noise(seed, **kwargs):
                np.random.seed(seed)
                return np.random.laplace(kwargs["noise_mean"], kwargs["noise_std"])

        else:
            raise NotImplementedError("noise_type must be one of ['gaussian', 'laplace']")

        super().__init__(
            generate_node_functional_parameters=get_weight_node,
            generate_edge_functional_parameters=get_weight_edge,
            generate_noise_functional_parameters=get_noise_parameters,
            get_exogenous_noise=get_noise,
            get_covariate_from_parents=get_covariate_from_parents,
            get_covariate_from_parents_signature=get_covariate_from_parents_signature,
            **parent_kwargs,
        )
