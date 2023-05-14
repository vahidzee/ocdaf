"""
This file contains the synthetic datasets generated.
"""
from .base_dataset import OCDDataset
import typing as th
from .scm import SCMGenerator
import dypy
from lightning_toolbox import DataModule
import numpy as np

class SyntheticOCDDataset(OCDDataset):
    def __init__(
        self,
        observation_size: int,
        scm_generator: th.Union[SCMGenerator, str],
        scm_generator_args: th.Optional[th.Dict[str, th.Any]] = None,
        seed: th.Optional[int] = None,
        name: th.Optional[str] = None,
        enable_simulate: bool = True,
        standardization: bool = False,
        reject_outliers_n_far_from_mean: th.Optional[float] = None,
        # intervention
        intervention_nodes: th.Optional[th.List[th.Any]] = None,
        intervention_functions: th.Optional[th.List[th.Union[th.Callable, str, th.Dict, float]]] = None,
    ):
        """
        Args:
            scm_generator: an SCM Generator object or a string which contains the class directory
                that is a Pytorch Dataset object used.
            scm_generator_args: arguments to pass to the SCM Generator if it is a string
            seed: The seed which is used to simulate the data
            name: The name of the dataset
            enable_simulate: You can turn this off if you want to have a fast construction of the dataset with no simulation
            standardization: Whether to standardize the data or not
            intervention_nodes: The nodes to intervene on
            intervention_functions: A list of functions described as a dypy formatted string, a callable, or a float
                if it is a float, then it resembles a constant hard intervention. For soft interventions, use a callable
                that takes in the fullowing form:
                func(noises: np.array, parent_values: List[np.array], parent_parameters: List[Dict[str, Any]], node_parameters: Dict[str, Any]) -> np.array
        """
        scm_generator = (
            scm_generator
            if isinstance(scm_generator, SCMGenerator)
            else dypy.eval(scm_generator)(**scm_generator_args)
        )
        self.seed = seed
        self.scm = scm_generator.generate_scm()
        if enable_simulate:
            if intervention_functions is not None:
                if standardization:
                    raise Exception("Standardization is not supported when using interventions!")
                
                for i in range(len(intervention_functions)):
                    if isinstance(intervention_functions[i], str):
                        intervention_functions[i] = dypy.eval(intervention_functions[i])
                    elif isinstance(intervention_functions[i], dict):
                        intervention_functions[i] = dypy.eval_function(**intervention_functions[i])
                    elif isinstance(intervention_functions[i], float):
                        tmp = intervention_functions[i]
                        intervention_functions[i] = lambda a_, b_, c_, d_: tmp * np.ones_like(a_)
                        
            df = self.scm.simulate(observation_size, seed=self.seed,
                                   intervention_nodes=intervention_nodes,
                                   intervention_functions=intervention_functions)
        else:
            df = None
        super().__init__(
            samples=df, 
            dag=self.scm.dag, 
            name=name, 
            explanation=self.scm.get_description(), 
            standardization=standardization, 
            reject_outliers_n_far_from_mean=reject_outliers_n_far_from_mean,
            intervention_columns=intervention_nodes,
        )
