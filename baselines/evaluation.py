import typing as th
import numpy as np
import networkx as nx


def count_backward(perm: th.List[int], dag: np.array):
    """
    Args:
        perm (list): permutation of the nodes
        dag (np.array): adjacency matrix of the DAG

    Count the number of backward edges in a permutation

    This is used as a validation metric
    """

    def edge_exists(u: int, v: int) -> bool:
        return dag[u, v] if isinstance(dag, np.ndarray) else dag.has_edge(u, v)

    n = len(perm)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if edge_exists(perm[j], perm[i]):
                count += 1
    return count


def backward_relative_penalty(perm: th.List[int], dag: th.Union[np.array, nx.DiGraph]):
    """
    Args:
        perm (list): permutation of the nodes (or a list of permutations)
        dag (np.array): adjacency matrix of the DAG

    Compute the ratio of backward edges to all edges given a permutation
    If there are multiple permutations, compute the average score
    """

    def edge_exists(u: int, v: int) -> bool:
        return dag[u, v] if isinstance(dag, np.ndarray) else dag.has_edge(u, v)

    n = len(perm)
    backwards, all = 0, 0
    for i in range(n):
        for j in range(n):
            if edge_exists(perm[j], perm[i]):
                all += 1
                if i < j:
                    backwards += 1
    return 1.0 * backwards / all
