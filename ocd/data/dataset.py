import torch
import pandas as pd
import typing as th
import numpy as np
import networkx as nx


class OCDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: th.Union[pd.DataFrame, np.array],
        dag: th.Union[nx.DiGraph, np.array],
        intervention_column: th.Optional[int] = None,
        intervention_values: th.Optional[th.List[th.Any]] = None,
        name: th.Optional[str] = None,
    ):
        """
        Args:
            samples: a pd.DataFrame with the samples
            dag: a bnlearn DAG
            intervention_node: the node which has been intervened on (todo: add support for multiple nodes)
            intervention_values: the values of the intervened node (split equally and sequentially among the samples)
            name: name of the dataset (optional, used for printing)
        """
        self.name = name
        # sort columns of samples alphabetically
        self.samples = samples
        # set the intervention column
        self.intervention_column = intervention_column
        # intervention_values is the value that we intervened on
        self.intervention_values = intervention_values
        # set the dag
        self.dag = dag

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # return the idx-th sample, as a jnp.array
        return self.samples.iloc[idx].values

    def get_intervention_column(self, col_id: int) -> str:
        return self.samples.columns[col_id] if isinstance(self.samples, pd.DataFrame) else str(col_id)

    @property
    def is_interventional(self) -> bool:
        return self.intervention_column is not None

    def __repr__(self) -> str:
        if self.is_interventional:
            intervention_name = self.get_intervention_column(
                self.intervention_column)
            intervention = f", do({intervention_name}={self.intervention_values})"
        else:
            intervention = ""
        return (
            f"{self.__class__.__name__ if self.name is None else self.name.capitalize()}(n={len(self)}{intervention})"
        )
