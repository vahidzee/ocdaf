import torch
import pandas as pd
from typing import Union, Optional
import numpy as np
import networkx as nx

class OCDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        dag: nx.DiGraph,
        name: Optional[str] = None,
        standard: bool = False,
        reject_outliers: bool = False,
        outlier_threshold: float = 3.0,
    ):
        """
        Args:
            samples: a pandas DataFrame
            dag: a networkx DiGraph
            name: name of the dataset (optional, used for printing)
            standard: whether to standardize the samples or not, either ways self.samples_statistic
                            contains the mean and std of each column as a dictionary
            reject_outliers: whether to reject outliers or not, if True, then the samples that have an entry value with a z-score
                            greater than `threshold` or less than -`threshold` removed.
        """
        self.name = name
        # set the dag
        self.dag = dag
    
        if samples is None:
            return
        
        # standardize all the columns
        self.samples_statistics = {}
        # sort columns of samples alphabetically
        self.samples = samples.reindex(sorted(samples.columns), axis=1)
        
        # create an array outlier which is a false for all the rows initially
        outliers = np.zeros(len(self.samples), dtype=bool)
        
        for col in self.samples.columns:
            avg = self.samples[col].mean()
            std = self.samples[col].std()
            if standard:
                self.samples[col] = (self.samples[col] - avg) / (std + 1e-8)
            # for values in self.samples[col] compute the z-score 
            # and mark the rows with z-score greater than 3 or less than -3 as outliers
            if reject_outliers:
                outliers = outliers | (np.abs(self.samples[col] - avg) > outlier_threshold * std)
        
        # remove the outliers
        self.samples = self.samples.iloc[~outliers, :]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # return the idx-th sample, as a jnp.array
        ret = self.samples.iloc[idx].values
        # turn ret into float32
        ret = ret.astype(np.float32)
        return ret
