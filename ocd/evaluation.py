import typing as th
import numpy as np
import random
import networkx as nx


def dfs(node, adj, stack, visited):
    visited[node] = True
    # Create a permuted list of indices from 0 to len(adj)
    idx = list(range(len(adj)))
    random.shuffle(idx)
    for i in idx:
        if adj[node][i] and not visited[i]:
            dfs(i, adj, stack, visited)
    stack.append(node)


def topological_sort(adj):
    stack = []
    visited = [False] * len(adj)
    for i in range(len(adj)):
        if not visited[i]:
            dfs(i, adj, stack, visited)
    # invert the stack
    return stack[::-1]


def count_backward(perm: th.List[int], dag: np.array):
    """
    Args:
        perm (list): permutation of the nodes
        dag (np.array): adjacency matrix of the DAG

    Count the number of backward edges in a permutation

    This is used as a validation metric
    """
    n = len(perm)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if dag[perm[j], perm[i]]:
                count += 1
    return count


def backward_score(perm: th.Union[th.List[int], th.List[th.List[int]]], dag: th.Union[np.array, nx.DiGraph]):
    """
    Args:
        perm (list): permutation of the nodes (or a list of permutations)
        dag (np.array): adjacency matrix of the DAG

    Compute the ratio of backward edges to all edges given a permutation
    If there are multiple permutations, compute the average score
    """
    if isinstance(perm[0], list):
        return sum([backward_score(p, dag) for p in perm]) / len(perm)

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


def shd(dag1: np.array, dag2: np.array):
    """
    Compute the structural hamming distance (SHD) between two DAGs
    """
    return np.sum(dag1 != dag2)


def closure_distance(perm: th.List[int], dag: np.array):
    """
    Compare the closure of the DAG with the tournament induced by the permutation
    """
    n = len(perm)
    dag_closure = closure(dag)
    perm_tournament = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            perm_tournament[perm[i], perm[j]] = 1
    return shd(dag_closure, perm_tournament)


def closure(dag: np.array):
    for k in range(len(dag)):
        for i in range(len(dag)):
            for j in range(len(dag)):
                dag[i, j] = dag[i, j] or (dag[i, k] and dag[k, j])
    return dag
