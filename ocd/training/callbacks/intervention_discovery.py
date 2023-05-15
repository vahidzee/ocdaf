# a boiled down version of the original callback for interventional datasets
from lightning.pytorch.callbacks import Callback
from lightning_toolbox import TrainingModule
from lightning.pytorch import Trainer
import typing as th
import numpy as np
import networkx as nx
import ocd.evaluation as eval_metrics
from causallearn.search.ConstraintBased.PC import pc

import pandas as pd
from causallearn.utils.cit import kci, gsq
import networkx as nx


all_evaluation_metrics = {
    "backward_relative_penalty": eval_metrics.backward_relative_penalty,
    "count_backward": eval_metrics.count_backward,
    "posterior_parent_ratio": eval_metrics.posterior_parent_ratio,
}


class PermutationEvaluationCallback(Callback):
    def __init__(
        self,
        every_n_epochs: int = 1,
        on_maximization: bool = False, # only run on expectation
        num_samples: th.Optional[int] = None, # use batch_size instead
        evaluation_metrics: th.Optional[th.List[str]] = None,
    ):
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.on_maximization = on_maximization
        self.evaluation_metrics = {}
        if evaluation_metrics is not None:
            for metric_name in evaluation_metrics:
                self.evaluation_metrics[metric_name] = all_evaluation_metrics[metric_name]
        else:
            self.evaluation_metrics = all_evaluation_metrics
        
    def _get_res_dict(self, pl_module: TrainingModule):
        # save the results
        perm_model = pl_module.model.permutation_model

        all_permutations = perm_model(
            self.num_samples, return_matrix=False, permutation_type="hard", training_module=pl_module
        )["perm_mat"]

        permutation_map = {}
        unique_permutations, counts = np.unique(all_permutations, axis=0, return_counts=True)
        mx = None
        for perm, c in zip(unique_permutations, counts):
            key = "-".join([str(i) for i in perm])
            permutation_map[key] = 1.0 * c / np.sum(counts)
            if mx is None or permutation_map[key] > permutation_map[mx]:
                mx = key

        best_permutation = [int(i) for i in mx.split("-")]
        ret = {}
        ret["metrics"] = {"average": {}, "majority": {}}

        # Calculate all the metrics
        for metric_name, metric_func in self.evaluation_metrics.items():
            sm = 0
            running_avg = 0
            for perm, c in permutation_map.items():
                perm_int = [int(i) for i in perm.split("-")]
                score = metric_func(perm_int, self.causal_graph)
                running_avg = (running_avg * sm + score * c) / (sm + c)
                sm += c
            ret["metrics"]["average"][metric_name] = running_avg
            ret["metrics"]["majority"][metric_name] = metric_func(perm=best_permutation, dag=self.causal_graph)

        ret["permutation_map"], ret["most_common_permutation"] = permutation_map, mx

        return ret

    def _log_results(self, pl_module: TrainingModule) -> None:
        ret = self._get_res_dict(pl_module)
        for key1, val1 in ret["metrics"].items():
            for key2, val2 in val1.items():
                pl_module.log(f"metrics/{key1}-{key2}", float(val2))

    def on_fit_start(self, trainer: Trainer, pl_module: TrainingModule) -> None:
        data = trainer.datamodule.data.data
        n = data.shape[-1]
        self.num_samples = self.num_samples if self.num_samples is not None else trainer.datamodule.train_batch_size
        self.causal_graph = nx.DiGraph()
        # create a chain graph of size n
        for i in range(n):
            self.causal_graph.add_node(i)
            if i > 0:
                self.causal_graph.add_edge(i-1, i)
        print("Causal graph: ", self.causal_graph.edges)
        return super().on_fit_start(trainer, pl_module)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: TrainingModule) -> None:
        if not self.on_maximization and (
            getattr(pl_module.model, "current_phase", "maximization") == "maximization"):
            return super().on_train_epoch_end(trainer, pl_module)
        if self.every_n_epochs and trainer.current_epoch % self.every_n_epochs == 0:
            self._log_results(pl_module)
        return super().on_train_epoch_end(trainer, pl_module)
