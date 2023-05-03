from base import AbstractBaseline
from cdt.causality.graph import CAM as CDT_CAM
import networkx as nx
import typing as th


class CAM(AbstractBaseline):
    """CAM baseline from CDT"""

    def __init__(
        self,
        dataset: th.Union["OCDDataset", str],  # type: ignore
        dataset_args: th.Optional[th.Dict[str, th.Any]] = None,
        # hyperparameters
        linear: bool = False,
    ):
        super().__init__(dataset=dataset, dataset_args=dataset_args, name='CAM')
        self.linear = linear

    def estimate_order(self):
        samples = self.get_data(conversion="pandas")
        graph = CDT_CAM(score="linear" if self.linear else "nonlinear", pruning=False).predict(samples)
        orders = list(nx.topological_sort(graph))
        return orders
