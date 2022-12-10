import torch
from .utils import generate_interventions, get_bnlearn_dag
import pandas as pd
import typing as th
import numpy as np


class CausalDataset(torch.utils.data.Dataset):
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
        self.name = name
        # sort columns of samples alphabetically
        self.samples = samples.reindex(sorted(samples.columns), axis=1)

        # intervention_node is the node that we intervened on
        self.intervention_node = intervention_node
        # intervention_values is the value that we intervened on
        self.intervention_values = intervention_values

        # self.features is a list of the nodes in the DAG
        all_features_values = [x.variable_card for x in dag["model"].cpds]
        all_features = [x.variable for x in dag["model"].cpds]
        # create a mapping dictionary from self.features to self.feature_values
        self.feature_values_dict = dict(zip(all_features, all_features_values))

        # Create the self.feature_values that corresponds to the category sizes
        # and create self.featrues that corresponds to the category names
        self.features_values = []
        self.features = []
        for x in self.samples.columns:
            self.features.append(x)
            self.features_values.append(self.feature_values_dict[x])

        # assign self.dag
        self.dag = np.zeros((len(self.features), len(self.features)))
        for i, x in enumerate(self.samples.columns):
            for j, y in enumerate(self.samples.columns):
                if dag["adjmat"][x][y]:
                    self.dag[j][i] = 1

    def get_category_size(self, category_name: str) -> int:
        return self.feature_values_dict[category_name]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # return the idx-th sample, as a jnp.array
        return self.samples.iloc[idx].values

    def __repr__(self) -> str:
        intervention = (
            f", do({self.intervention_node}={self.intervention_values})" if self.intervention_node is not None else ""
        )
        return (
            f"{self.__class__.__name__ if self.name is None else self.name.capitalize()}(n={len(self)}{intervention})"
        )

    def get_adjacency_matrix(self):
        # a pd.DataFrame
        adjmat = self.dag["adjmat"] if self.dag is not None else None
        # adjmat is a pd.DataFrame, with the nodes as index and columns and the values are 1 if there is an edge, 0 otherwise
        # convert it into a numpy array and return it
        return adjmat.values


def generate_datasets(
    name,
    observation_size,
    intervention_size=0,
    node_list=None,
    dag=None,
    import_configs=None,
    show_progress=False,
    seed=0,
) -> th.List[CausalDataset]:
    """
    Generate a list of datasets, first dataset is the original dataset, the rest are interventions

    Args:
        name (str): name of the dataset (or link to the dataset)
        dag (bnlearn.BayesianModel): the DAG
        observation_size (int): number of samples in the original dataset (observational data)
        intervention_size (int): number of samples in each intervention (per value) dataset (default: 0)
        node_list (list): list of nodes to intervene on (default is all nodes)
        import_configs (dict): configs to pass to bnlearn.import_DAG
        show_progress (bool): show progress bar for sampling
        seed (int): seed for sampling (default: 0)

    Returns:
        list: list of datasets (CausalDataset) (first dataset is the original dataset, the rest are interventions)
    """
    # generate observational data
    dag = dag if dag is not None else get_bnlearn_dag(name, import_configs=import_configs)
    # get last slug of the name, remove the extension
    name = name.split("/")[-1].split(".")[0]

    datasets = [
        CausalDataset(
            name=name, samples=dag["model"].simulate(observation_size, show_progress=show_progress, seed=seed), dag=dag
        )
    ]
    if intervention_size == 0:
        return datasets
    # generate interventional data
    interventions = generate_interventions(
        dag=dag,
        num_samples_per_value=intervention_size,
        node_list=node_list,
        show_progress=show_progress,
        seed=seed + 1,  # seed for the interventions
    )
    for node_intervention in interventions:
        # merge sample dataframes into one dataframe
        samples = pd.concat([samples for _, _, samples in node_intervention])
        # create a dataset
        datasets.append(
            CausalDataset(
                dag=dag,
                name=name,
                samples=samples,
                intervention_node=node_intervention[0][0],
                intervention_values=[value for _, value, _ in node_intervention],
            )
        )
    return datasets
