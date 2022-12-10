import typing as th
import random


def args_list_len(*args):
    return max(len(arg) if isinstance(arg, (tuple, list)) else (1 if arg is not None else 0) for arg in args)


def list_args(*args, length: th.Optional[int] = None, return_length: bool = False):
    length = args_list_len(*args) if length is None else length
    if not length:
        results = args if len(args) > 1 else args[0]
        return results if not return_length else (results, length)
    results = [([arg] * length if not isinstance(arg, (tuple, list)) else arg)
               for arg in args]
    results = results if len(args) > 1 else results[0]
    return results if not return_length else (results, length)


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
