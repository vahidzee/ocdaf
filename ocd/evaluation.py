
import typing as th
from ocd.data import generate_datasets
import numpy as np

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
        for j in range(i+1, n):
            if dag[perm[j], perm[i]]:
                count += 1
    return count

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
        for j in range(i+1, n):
            perm_tournament[perm[i], perm[j]] = 1
    return shd(dag_closure, perm_tournament)

def closure(dag: np.array):
    for k in range(len(dag)):
        for i in range(len(dag)):
            for j in range(len(dag)):
                dag[i, j] = dag[i, j] or (dag[i, k] and dag[k, j])
    return dag




    