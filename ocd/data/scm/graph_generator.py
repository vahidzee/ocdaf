from .base_generator import BaseGenerator
import typing as th
import numpy as np
import networkx as nx
import random


class GraphGenerator(BaseGenerator):
    def __init__(
        self,
        seed: th.Optional[int] = None,
        base_dag: th.Optional[np.array] = None,
        graph_type: th.Optional[
            th.Literal[
                "erdos_renyi",
                "barabasi_albert",
                "random_dag",
                "chain",
                "collider",
                "full",
                "tree",
                "fork",
                "v_structure",
            ]
        ] = None,
        enforce_ordering: th.Optional[th.List] = None,
        **kwargs
    ) -> None:
        super().__init__(seed=seed)
        self.base_dag = base_dag
        self.graph_generator_type = graph_type
        self.graph_generator_args = kwargs
        self.enforce_ordering = enforce_ordering

    def generate_dag(self) -> nx.DiGraph:
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

        # if the base graph is not given, generate the graph using a generator
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
            new_dag = nx.star_graph(self.graph_generator_args["n"] - 1)
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

        # permute the nodes of new_dag if the ordering is not enforced, otherwise, do it according to the ordering
        correct_ordering = nx.topological_sort(new_dag)

        np.random.seed(seed=new_seed)
        random.seed(new_seed)
        if self.enforce_ordering is not None:
            new_dag = nx.relabel_nodes(new_dag, dict(zip(correct_ordering, self.enforce_ordering)))
        else:
            mapping = {}
            all_in_order = []
            for x in correct_ordering:
                all_in_order.append(x)
            all_2 = all_in_order.copy()
            random.shuffle(all_2)
            for x, y in zip(all_in_order, all_2):
                mapping[x] = y
            new_dag = nx.relabel_nodes(new_dag, mapping)
        return new_dag
