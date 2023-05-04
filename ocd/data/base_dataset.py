import torch
import pandas as pd
import typing as th
import numpy as np
import networkx as nx

class OCDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: th.Optional[th.Union[pd.DataFrame, np.array]],
        dag: nx.DiGraph,
        intervention_column: th.Optional[int] = None,
        intervention_values: th.Optional[th.List[th.Any]] = None,
        name: th.Optional[str] = None,
        explanation: th.Optional[str] = None,
        standardization: bool = True,
    ):
        """
        Args:
            samples: a pd.DataFrame with the samples
            dag: a networkx DiGraph or a np.array with the DAG
            intervention_node: the node which has been intervened on (todo: add support for multiple nodes)
            intervention_values: the values of the intervened node (split equally and sequentially among the samples)
            name: name of the dataset (optional, used for printing)
            standardization: whether to standardize the samples or not, either ways self.samples_statistic
                            contains the mean and std of each column as a dictionary
        """
        self.name = name
        self.explanation = explanation
        
        # set the intervention column
        self.intervention_column = intervention_column
        # intervention_values is the value that we intervened on
        self.intervention_values = intervention_values
        # set the dag
        self.dag = dag
        
        
        if samples is None:
            return
        
        # standardize all the columns
        self.samples_statistics = {}
        # sort columns of samples alphabetically
        self.samples = samples.reindex(sorted(samples.columns), axis=1)
        
        if isinstance(self.samples, pd.DataFrame):
            for col in self.samples.columns:
                avg = self.samples[col].mean()
                std = self.samples[col].std()
                self.samples_statistics[col] = {'mean': avg, 'std': std}
                if standardization:
                    self.samples[col] = (self.samples[col] - avg) / std
        elif isinstance(self.samples, np.array):
            for col in range(self.samples.shape[1]):
                avg = np.mean(self.samples[:, col])
                std = np.std(self.samples[:, col])
                self.samples_statistics[col] = {'mean': avg, 'std': std}
                if standardization:
                    self.samples[:, col] = (self.samples[:, col] - avg) / std
        else:
            raise ValueError("samples must be either a pd.DataFrame or a np.array")
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # return the idx-th sample, as a jnp.array
        ret = self.samples.iloc[idx].values
        # turn ret into float32
        ret = ret.astype(np.float32)
        return ret

    def get_intervention_column(self, col_id: int) -> str:
        return self.samples.columns[col_id] if isinstance(self.samples, pd.DataFrame) else str(col_id)

    @property
    def is_interventional(self) -> bool:
        return self.intervention_column is not None

    def __repr__(self) -> str:
        if self.is_interventional:
            intervention_name = self.get_intervention_column(self.intervention_column)
            intervention = f", do({intervention_name}={self.intervention_values})"
        else:
            intervention = ""
        return (
            f"{self.__class__.__name__ if self.name is None else self.name.capitalize()}(n={len(self)}{intervention})"
        )
