"""
This file contains two classes that are used to generate data for the SCM.

- The first class is a simple DAG generator that can be used to generate a DAG. This
DAG is then used to generate data using the SCM.

- The second class is an SCM generator that gets input the DAG generator as well.
After generating a DAG, it uses a specific scheme to generate the functional parameters for the
SCM. The SCM generator class is abstract, which means some functions of it should be implemented. Namely,
the functions that generate the SCM parameters that govern the data generation process. For more information,
check out the description of the specific functions and override them in a child class that you feel fit.
"""

import networkx as nx
import typing as th
from .base_scm import SCM
from abc import ABC, abstractmethod
from .base_generator import BaseGenerator
from ocd.data.synthetic.graph_generator import GraphGenerator
import numpy as np

class SCMGenerator(ABC, BaseGenerator):
    def __init__(
        self,
        # Graph generator parameters or construction
        graph_generator: th.Union[GraphGenerator, str],
        graph_generator_args: th.Optional[th.Dict[str, th.Any]] = None,
        # seed
        seed=None,
    ):
        """
        The SCM generator has two parts, the first one generates a Graph which is given via the graph_generator
        object. The second part is defining relevant functions that calculate each covariate from the parents.

        To create these function, we must "parametrize" them. The function_generator_args is a dictionary that
        contains all which is needed to parametrize the functions. Now to come up with parameters for each function,
        we use a scheme where we put parameters on top of each edge and node and noise variable. This is done in a purely
        generic fashion whereby we have three functions that are abstract and should be implemented:

        (1) generate_edge_functional_parameters:
            For each edge return a dictionary of parameters that can be used later on for the data generating process.

        (2) generate_node_functional_parameters:
            For each node return a dictionary of parameters that can be used later on for the data generating process.

        (3) generate_noise_functional_parameters:
            For each noise variable return a dictionary of parameters that can be used later on for the data generating process.


        When the generate_scm function is called, it will call the above functions for each edge, node and noise variable.
        Now, we will need to also implement the functions that use these parameters to generate noise values and also
        use these to link the parents to the child. This is done in the following two functions:

        (1) get_exogenous_noise:
        This function returns a number that is the noise value associated with the specific node.

        (2) get_covariate_from_parents:
        This function calculates the covariate value for the specific node given the parents.

        Last but not least, for the sake of convenience, we also have the following function that can be used to
        print out the whole SCM. The function "get_covariate_from_parents_signature" is used to print out a formula
        that related the parents to the covariate. This is also an abstract function that should be implemented.
        """
        BaseGenerator.__init__(self, seed=seed)
        self.graph_generator = (
            graph_generator
            if isinstance(graph_generator, GraphGenerator)
            else dypy.eval(graph_generator)(**graph_generator_args)
        )
        self.sample_count = 0

    @abstractmethod
    def generate_edge_functional_parameters(self, dag: nx.DiGraph, child: int, par: int, seed: int):
        """
        For each edge return a dictionary of parameters that can be used later on for the data generating process.
        For example, if one wants to save a 'weight' parameter for each edge and has the following information available
        in the "function_generator_args":
        {
            edge_weight_lo: 0.5
            edge_weight_hi: 1.5
        }
        In which case, the weight value will come from a Uniform(0.5, 1.5) distribution and the returned dictionary
        would be as follows:
        {
            weight: 0.6 (an example sampled from Unif[0.5, 1.5])
        }
        """
        raise NotImplementedError("generate_edge_functional_parameters must be implemented")

    @abstractmethod
    def generate_noise_functional_parameters(self, dag: nx.DiGraph, node: int, seed: int) -> th.Dict[str, th.Any]:
        """
        For the noise in every node we might have parameters. This function returns a dictionary of such parameters.
        For example, if we aim to save the standard deviation and mean of the noise values coming from a Gaussian distribution,
        we may have the palette for such parameters in the "function_generator_args":
        {
            std_lo: 0.5,
            std_hi: 1.5,
            mean_lo: -1
            mean_hi: +1
        }
        In which case, when the SCM is generated, each noise will be a "Guassian(mean, std)" with
        mean ~ Unif(mean_lo, mean_hi)
        std ~ Unif(std_lo, std_hi)
        In which case, the following will be the returned dictionary:
        {
            mean: 0.1 (an example sampled from Unif[-1, 1])
            std: 0.6 (an example sampled from Unif[0.5, 1.5])
        }

        Args:
            dag: The DAG that the SCM is based on
            node: The node that the noise is associated with
            seed: int

        Returns:
            A dictionary of parameters that can be used later on for the data generating process.
        """
        raise NotImplementedError("generate_noise_functional_parameters must be implemented")

    @abstractmethod
    def generate_node_functional_parameters(self, dag: nx.DiGraph, node: int, seed: int) -> th.Dict[str, th.Any]:
        """
        For each node return a dictionary of parameters that can be used later on for the data generating process.
        For example, if one wants to save a 'weight' parameter for each node and has the following information available
        in the "function_generator_args":
        {
            node_weight_lo: 0.5
            node_weight_hi: 1.5
        }
        In which case, the weight value will come from a Uniform(0.5, 1.5) distribution and the returned dictionary
        would be as follows:
        {
            weight: 0.6 (an example sampled from Unif[0.5, 1.5])
        }

        Args:
            dag: The DAG that the SCM is based on
            node: The node that the node parameters are associated with
            seed: int

        Returns:
            A dictionary of parameters that can be used later on for the data generating process.
        """
        raise NotImplementedError("generate_node_functional_parameters must be implemented")

    @abstractmethod
    def get_exogenous_noise(self, noise_parameters: th.Dict[str, th.Any], seed: int, n_samples: int) -> np.array:
        """
        Return the noise value for the specific node given the parameters for that node.

        For example, if the noise is a Gaussian distribution with mean and std as parameters, the function
        will return a sample from a Gaussian(mean, std) distribution.
        """
        raise NotImplementedError("get_exogenous_noise must be implemented")

    @abstractmethod
    def get_covariate_from_parents(
        self,
        noise: np.array,
        parents: th.List[np.array],
        parent_parameters: th.List[th.Dict[str, th.Any]],
        node_parameters: th.List[th.Dict[str, th.Any]],
    ) -> np.array:
        """
        Return the covariate value based on the noise on that node, the parents and the parameters of the parents

        For example, if the covariate is a linear combination plus a non-linearity (Sigmoid)of the parents with the
        weight parameter of each edge corresponding to the coefficient of that parent and the weight parameter in the
        node corresponding to the bias, the function will return the linear combination of the parents plus the bias:

        covariate = sigmoid(sum(parents * parent_parameters[weight]) + node_parameters[weight]) + noise

        Args:
            noise: The noise value for the node
            parents: The values of the parent covariates as a list of floats
            parent_parameters: The parameters of the parents as a list of dictionaries
            node_parameters: The parameters of the node as a dictionary
        Returns:
            The value of the covariate calculated based on the parents and the parameters of the parents
            and the node parameters and noise
        """
        raise NotImplementedError("get_covariate_from_parents must be implemented")

    @abstractmethod
    def get_covariate_from_parents_signature(
        self,
        node: int,
        parents: th.List[int],
        node_parameters: th.Dict[str, th.Any],
        noise_parameters: th.Dict[str, th.Any],
        parent_parameters: th.List[th.Dict[str, th.Any]],
    ) -> str:
        """
        This function gets the list of parent parameters, node parameters, and noise parameters.
        It also gets the list of the actual parents and returns a string which is the signature of the function
        that is relating the parents to the node.

        For example, the function might return the following string:

        x(2) = sigmoid(w(2, 1) * x(1) + w(2, 3) * x(3) + b(2)) + noise(2)

        This function is created only for the purpose of visualization.

        Args:
            node: The node that the node parameters are associated with
            paranets: The parent nodes in a list
            node_parameters: The parameters of the node as a dictionary
            noise_parameters: The parameters of the noise as a dictionary
            parent_parameters: The parameters of the parents as a list of dictionaries
        Returns:
            A string representing a visualization of the function that is relating the parents to the node
        """
        raise NotImplementedError("get_covariate_from_parents_signature must be implemented")

    def generate_scm(self) -> SCM:
        """
        Generates an SCM based on the graph generator parameters.

        Returns:
            SCM: an SCM
        """

        dag = self.graph_generator.generate_dag()

        noise_parameters = {}
        node_parameters = {}
        parent_parameters = {}
        parents = {}
        # add generative parameters to the nodes of the dag

        for node in dag.nodes:
            seed = self.get_iterative_seed()
            noise_parameters[node] = self.generate_noise_functional_parameters(dag, node, seed=seed)
            seed = self.get_iterative_seed()
            node_parameters[node] = self.generate_node_functional_parameters(dag, node, seed=seed)
            parent_parameters[node] = []
            parents[node] = []
            for par in dag.predecessors(node):
                seed = self.get_iterative_seed()
                parent_parameters[node].append(self.generate_edge_functional_parameters(dag, node, par, seed=seed))
                parents[node].append(par)

        return SCM(
            dag,
            noise_parameters,
            node_parameters,
            parent_parameters,
            parents,
            get_exogenous_noise=self.get_exogenous_noise,
            get_covariate_from_parents=self.get_covariate_from_parents,
            get_covariate_from_parents_signature=self.get_covariate_from_parents_signature,
        )
