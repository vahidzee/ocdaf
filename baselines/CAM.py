from baselines.baseline import Baseline
from ocd.data import OCDDataset
from cdt.causality.graph import CAM as CDT_CAM
import networkx as nx


class CAM(Baseline):
    def __init__(self):
        super().__init__("CAM")

    def fit(self, dataset: OCDDataset, linear: bool = False):
        """ Fit the baseline on the dataset
        Args:
            dataset: The dataset to fit on
            linear: If the data is linear
        """
        samples = dataset.samples
        score = 'linear' if linear else 'nonlinear'
        graph = CDT_CAM(score=score, pruning=False).predict(samples)
        orders = list(nx.topological_sort(graph))
        return orders
