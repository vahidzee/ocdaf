import torch
from .utils import generate_interventions, get_bnlearn_dag
import pandas as pd
import typing as th
import numpy as np
from ..dataset import OCDDataset


class BNLearnOCDDataset(OCDDataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        dag,
        intervention_node: th.Optional[str] = None,
        intervention_values: th.Optional[th.List[th.Any]] = None,
        name: th.Optional[str] = None,
    ):
        """
        A wrapper around samples from bnlearn DAGs.

        Args:
            samples: a pd.DataFrame with the samples
            dag: a bnlearn DAG
            intervention_node: the node which has been intervened on (todo: add support for multiple nodes)
            intervention_values: the values of the intervened node (split equally and sequentially among the samples)
            name: name of the dataset (optional, used for printing)
        """
        # sort columns of samples alphabetically and set the intervention column
        samples = samples.reindex(sorted(samples.columns), axis=1)
        intervention_column = samples.columns.get_loc(
            intervention_node) if intervention_node is not None else None

        # self.features is a list of the nodes in the DAG
        all_features_values = [x.variable_card for x in dag["model"].cpds]
        all_features = [x.variable for x in dag["model"].cpds]
        # create a mapping dictionary from self.features to self.feature_values
        self.feature_values_dict = dict(zip(all_features, all_features_values))

        # Create the self.feature_values that corresponds to the category sizes
        # and create self.featrues that corresponds to the category names
        self.features_values = []
        self.features = []
        for x in samples.columns:
            self.features.append(x)
            self.features_values.append(self.feature_values_dict[x])
        # assign self.dag
        new_dag = np.zeros((len(self.features), len(self.features)))
        for i, x in enumerate(samples.columns):
            for j, y in enumerate(samples.columns):
                if dag["adjmat"][x][y]:
                    new_dag[j][i] = 1

        super().__init__(
            samples=samples,
            dag=new_dag,
            intervention_column=intervention_column,
            intervention_values=intervention_values,
            name=name
        )
