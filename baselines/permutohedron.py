from daguerreo.models import Daguerro
from daguerreo import utils
from daguerreo.args import parse_pipeline_args
import networkx as nx
import typing as th

from base import AbstractBaseline


class Permutohedron(AbstractBaseline):
    """DAG Learning on the Permutohedron baseline. https://arxiv.org/pdf/2301.11898.pdf"""

    def __init__(
            self,
            dataset: th.Union["OCDDataset", str],  # type: ignore
            name: th.Optional[str] = None,  # name to override the default name
            dataset_args: th.Optional[th.Dict[str, th.Any]] = None,
            # hyperparameters
            linear: bool = False,
            seed: int = 42,
    ):
        super().__init__(dataset=dataset, dataset_args=dataset_args, name=name)
        self.linear = linear

        # parse args
        arg_parser = parse_pipeline_args()
        self.args = arg_parser.parse_args()
        self.seed = seed
        self.args.standardize = True
        self.args.sparsifier = 'none'
        self.args.equations = 'linear' if self.linear else 'nonlinear'

    def estimate_order(self):
        utils.init_seeds(seed=self.seed)
        samples = self.get_data(conversion="pandas")
        daguerro = Daguerro.initialize(samples, self.args, self.args.joint)
        daguerro, samples = utils.maybe_gpu(self.args, daguerro, samples)
        log_dict = daguerro(samples, utils.AVAILABLE[self.args.loss], self.args)
        daguerro.eval()
        _, dags = daguerro(samples, utils.AVAILABLE[self.args.loss], self.args)

        estimated_adj = dags[0].detach().cpu().numpy()
        g = nx.DiGraph(estimated_adj)
        orders = list(nx.topological_sort(g))
        return orders
