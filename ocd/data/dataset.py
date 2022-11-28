import torch
from .utils import generate_interventions, get_bnlearn_dag
import pandas as pd
import typing as th


class CausalDataset(torch.utils.data.Dataset):
    def __init__(
        self, samples: th.Any, dag, intervention_node=None, intervention_values=None, name: th.Optional[str] = None
    ):
        self.name = name
        self.samples = samples

        # intervention_node is the node that we intervened on
        self.intervention_node = intervention_node
        # intervention_values is the value that we intervened on
        self.intervention_values = intervention_values

        self.dag = dag
        self.features = list(samples.columns) if isinstance(samples, pd.DataFrame) else None
        self.features_values = [dag["model"].get_cpds(node).values.shape[0] for node in dag["model"].nodes()]

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
        adjmat= self.dag["adjmat"] if self.dag is not None else None # a pd.DataFrame
        # adjmat is a pd.DataFrame, with the nodes as index and columns and the values are 1 if there is an edge, 0 otherwise
        # convert it into a numpy array and return it
        return adjmat.values




def generate_datasets(
    name, observation_size, intervention_size, node_list=None, dag=None, import_configs=None, show_progress=False
):
    """
    Generate a list of datasets, first dataset is the original dataset, the rest are interventions

    Args:
        name (str): name of the dataset (or link to the dataset)
        dag (bnlearn.BayesianModel): the DAG
        observation_size (int): number of samples in the original dataset (observational data)
        intervention_size (int): number of samples in each intervention (per value) dataset
        node_list (list): list of nodes to intervene on (default is all nodes)
        import_configs (dict): configs to pass to bnlearn.import_DAG
        show_progress (bool): show progress bar for sampling

    Returns:
        list: list of datasets (CausalDataset) (first dataset is the original dataset, the rest are interventions)
    """
    # generate observational data
    dag = dag if dag is not None else get_bnlearn_dag(name, import_configs=import_configs)
    # get last slug of the name, remove the extension
    name = name.split("/")[-1].split(".")[0]

    datasets = [
        CausalDataset(name=name, samples=dag["model"].simulate(observation_size, show_progress=show_progress), dag=dag)
    ]
    # generate interventional data
    interventions = generate_interventions(
        dag=dag, num_samples_per_value=intervention_size, node_list=node_list, show_progress=show_progress
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
