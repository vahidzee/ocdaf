from math import inf
from .dataset import OCDDataset
import networkx as nx
import numpy as np
import typing as th
import dycode
import pandas as pd


class SCM:
    """
    A generic structural causal model
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        functional_parameters: th.Dict[th.Any, th.List[th.Any]],
        get_exogenous_noise: th.Optional[th.Callable] = None,
        get_covariate_from_parents: th.Optional[th.Callable] = None,
        get_covariate_from_parents_signature: th.Optional[th.Callable] = None,
    ):
        """
        Args:
            dag: a networkx.DiGraph
            functional_parameters: a dictionary of parameters for each node
                [**noise_parameters, **node_parameters, **par1_parameters, par1, **par2_parameters, par2, ...]
        """
        self.dag = dag
        self.functional_parameters = functional_parameters
        self.topological_order = list(nx.topological_sort(dag))

        self.get_exogenous_noise = get_exogenous_noise
        self.get_covariate_from_parents = get_covariate_from_parents
        self.get_covariate_from_parents_signature = get_covariate_from_parents_signature

    def simulate(
        self,
        n_samples,
        seed=None,
        intervention_node: th.Optional[th.Any] = None,
        intervention_function: th.Optional[th.Callable] = None,
    ) -> OCDDataset:
        vals = {x: [] for x in self.dag.nodes}
        for sample_i in range(n_samples):
            for v in self.topological_order:
                # sample exogenous noise
                noise = self.get_exogenous_noise(seed + sample_i, **self.functional_parameters[v][0])
                inputs = [noise]
                params = [self.functional_parameters[v][1]]
                for i in range(2, len(self.functional_parameters[v]), 2):
                    params.append(self.functional_parameters[v][i])
                    inputs.append(vals[self.functional_parameters[v][i + 1]][-1])
                if v == intervention_node:
                    vals[v].append(intervention_function(inputs, params))
                else:
                    vals[v].append(self.get_covariate_from_parents(inputs, params))
        samples = pd.DataFrame(vals)
        return samples.reindex(sorted(samples.columns), axis=1)

    def nodes(self):
        return self.dag.nodes

    def draw(self, with_labels=False):
        # take a copy of self.dag
        dag = self.dag.copy()

        node_labels = {}
        for v in self.topological_order:
            inputs = [self.functional_parameters[v][0]]
            params = [self.functional_parameters[v][1]]
            for i in range(2, len(self.functional_parameters[v]), 2):
                params.append(self.functional_parameters[v][i])
                inputs.append(self.functional_parameters[v][i + 1])
            node_labels[v] = (
                self.get_covariate_from_parents_signature(inputs, params)
                if self.get_covariate_from_parents_signature is not None
                else None
            )

            node_labels[v] = f"x{(v)}" if node_labels[v] is None else f"x({v}) = {node_labels[v]}"

        # get node positions from networkx
        pos = nx.drawing.layout.spring_layout(dag)

        nx.draw(dag, pos=pos, with_labels=not with_labels)
        if with_labels:
            nx.draw_networkx_labels(dag, pos=pos, labels=node_labels, font_size=8)
        else:
            # print all the values in node_labels
            for k, v in node_labels.items():
                print(f"{v}\n---------------------\n")

    def count_backward(self, ordering: th.List[int]) -> int:
        """
        If we apply ordering to the graph of self.dag, how many edges will be reversed?
        """
        # get the index of each node in ordering
        ordering = {v: i for i, v in enumerate(ordering)}
        # count the number of edges that will be reversed
        count = 0
        for u, v in self.dag.edges:
            if ordering[u] > ordering[v]:
                count += 1
        return count


GRAPH_GENERATOR_TYPE = [
    "erdos_renyi",
    "barabasi_albert",
    "random_dag",
    "chain",
    "collider",
    "full",
    "tree",
    "fork",
    "v_structure",
    None,
]


class SCMGenerator:
    def __init__(
        self,
        # Dycode parameters for dynamic setup of causal models from the datamodule itself
        generate_node_functional_parameters: th.Callable,
        generate_edge_functional_parameters: th.Callable,
        generate_noise_functional_parameters: th.Callable,
        get_exogenous_noise: th.Callable,
        get_covariate_from_parents: th.Callable,
        get_covariate_from_parents_signature: th.Optional[th.Callable] = None,
        # Graph generator parameters
        base_dag: th.Optional[np.array] = None,
        graph_generator_type: GRAPH_GENERATOR_TYPE = None,
        graph_generator_args: th.Optional[th.Dict[str, th.Any]] = None,
        # functional parameters
        functional_form_generator_args: th.Optional[th.Dict[str, th.Any]] = None,
        # seed
        seed=None,
    ):
        self.seed = seed
        self.base_dag = base_dag
        self.graph_generator_type = graph_generator_type
        self.graph_generator_args = graph_generator_args or {}

        self.functional_form_generator_args = functional_form_generator_args or {}

        self.sample_count = 0

        self.generate_node_functional_parameters = generate_node_functional_parameters
        self.generate_edge_functional_parameters = generate_edge_functional_parameters
        self.generate_noise_functional_parameters = generate_noise_functional_parameters

        self.get_exogenous_noise = get_exogenous_noise
        self.get_covariate_from_parents = get_covariate_from_parents
        self.get_covariate_from_parents_signature = get_covariate_from_parents_signature

    def get_iterative_seed(self):
        """
        Returns:
            int: a new seed based on the current seed and sample count and increases the sample count
        """
        self.sample_count += 1
        return self.seed + self.sample_count

    def generate_dag(self) -> np.array:
        """
        If base_dag is given, then create a dag based on that.
        Otherwise, generate a random graph using the specified graph generator
        and then make it Acyclic via a random permutation.

        Returns:
            nx.Graph: a random graph
        """
        new_seed = self.get_iterative_seed()
        # create a nx.Graph using base_dag as the adjacency matrix
        if self.base_dag is not None:
            new_dag = nx.DiGraph(self.base_dag)
            return new_dag

        if self.graph_generator_type == "erdos_renyi":
            # generate a random graph using erdos renyi model
            new_dag = nx.gnp_random_graph(
                self.graph_generator_args["n"], self.graph_generator_args["p"], seed=new_seed
            )
        elif self.graph_generator_type == "barabasi_albert":
            # generate a random graph using barabasi albert model
            new_dag = nx.barabasi_albert_graph(
                self.graph_generator_args["n"], self.graph_generator_args["m"], seed=new_seed
            )
        elif self.graph_generator_type == "random_dag":
            # generate a random graph
            new_dag = nx.gnm_random_graph(
                self.graph_generator_args["n"], self.graph_generator_args["m"], seed=new_seed
            )
        elif self.graph_generator_type == "chain":
            # generate a chain graph
            new_dag = nx.path_graph(self.graph_generator_args["n"])
        elif self.graph_generator_type in ["collider", "fork", "v_structure"]:
            # generate a star graph
            new_dag = nx.star_graph(self.graph_generator_args["n"])
            # swap the first and last node
            if self.graph_generator_type in ["collider", "v_structure"]:
                all_nodes = list(new_dag.nodes.keys())
                new_dag = nx.relabel_nodes(new_dag, {all_nodes[0]: all_nodes[-1], all_nodes[-1]: all_nodes[0]})
        elif self.graph_generator_type == "full":
            # generate a complete graph
            new_dag = nx.complete_graph(self.graph_generator_args["n"])
        elif self.graph_generator_type == "tree":
            # generate a jungle graph
            new_dag = nx.random_tree(self.graph_generator_args["n"], seed=new_seed)

        else:
            raise ValueError("dag_generator_args must be specified when base_dag is not provided")
        # make new_dag directed acyclic graph

        # create a directed graph from new_dag
        new_dag = nx.DiGraph(new_dag)

        # iterate over all the edges of new_dag
        delete_edges = []
        for edge in new_dag.edges:
            # check if the edge is directed from a node with a higher index to a node with a lower index
            # get the location of edge[0] and edge[1] in the permutation
            u = edge[0] if isinstance(edge[0], int) else list(new_dag.nodes).index(edge[0])
            v = edge[1] if isinstance(edge[1], int) else list(new_dag.nodes).index(edge[1])
            if u >= v:
                # remove every such edge
                delete_edges.append((edge[0], edge[1]))

        new_dag.remove_edges_from(delete_edges)

        # permute the nodes of new_dag
        np.random.seed(seed=new_seed)
        new_dag = nx.relabel_nodes(new_dag, dict(zip(new_dag.nodes, np.random.permutation(new_dag.nodes))))
        return new_dag

    def generate_scm(self) -> SCM:
        """
        Generates a SCM based on the graph generator parameters.

        Returns:
            SCM: a SCM
        """
        dag = self.generate_dag()
        functional_parameters = {}
        # add generative parameters to the nodes of the dag

        for node in dag.nodes:
            seed = self.get_iterative_seed()
            functional_parameters[node] = [
                self.generate_noise_functional_parameters(dag, node, seed=seed, **self.functional_form_generator_args)
            ]
            seed = self.get_iterative_seed()
            functional_parameters[node].append(
                self.generate_node_functional_parameters(dag, node, seed=seed, **self.functional_form_generator_args)
            )
            for par in dag.predecessors(node):
                seed = self.get_iterative_seed()
                functional_parameters[node].append(
                    self.generate_edge_functional_parameters(
                        dag, node, par, seed=seed, **self.functional_form_generator_args
                    )
                )
                functional_parameters[node].append(par)

        return SCM(
            dag,
            functional_parameters,
            get_exogenous_noise=self.get_exogenous_noise,
            get_covariate_from_parents=self.get_covariate_from_parents,
            get_covariate_from_parents_signature=self.get_covariate_from_parents_signature,
        )
