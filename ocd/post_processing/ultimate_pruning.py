import pandas as pd
import typing as th
from .pc_pruning import pc_based_pruning
from .cam_pruning import sparse_regression_based_pruning
import networkx as nx

def ultimate_pruning(df: pd.DataFrame, ordering: th.List, independence_test: th.Literal['kci', 'g2']='kci', verbose=True, alpha=0.05):
    # change the dataframe to a numpy array
    pruned_graph = pc_based_pruning(df, ordering, independence_test, verbose, alpha)
    
    # get the skeleton of pruned_graph
    skeleton = nx.Graph(pruned_graph)
    
    
    if verbose:
        print("Pruned graph after PC-based pruning and this is the skeleton:")
        for edge in skeleton.edges():
            print(edge)
        print("----")
        
    # find all the connected components of the skeleton
    connected_components = list(nx.connected_components(skeleton))
    
    if verbose:
        print("The connected components of the skeleton are:")
        for component in connected_components:
            print(component)
        print("----")
        
    ordering_input_to_cam = []
    for component in connected_components:
        comp = list(component)
        # order comp according to the ordering
        comp.sort(key=lambda x: ordering.index(x))
        ordering_input_to_cam += comp
    
    pruned_graph = sparse_regression_based_pruning(df, ordering_input_to_cam, verbose, cutoff=0.001)
    
    return pruned_graph
    