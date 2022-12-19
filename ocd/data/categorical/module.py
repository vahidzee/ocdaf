"""
This file contains the datamodule used for the categorical dataset.
"""

import typing as th
from .utils import generate_interventions, get_bnlearn_dag
from .bnlearn import BNLearnOCDDataset
from ..module import OCDDataModule
import pandas as pd


def generate_datasets(
    name,
    observation_size,
    intervention_size=0,
    node_list=None,
    dag=None,
    import_configs=None,
    show_progress=False,
    seed=0,
) -> th.List[BNLearnOCDDataset]:
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
    dag = dag if dag is not None else get_bnlearn_dag(
        name, import_configs=import_configs)
    # get last slug of the name, remove the extension
    name = name.split("/")[-1].split(".")[0]

    datasets = [
        BNLearnOCDDataset(
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
            BNLearnOCDDataset(
                dag=dag,
                name=name,
                samples=samples,
                intervention_node=node_intervention[0][0],
                intervention_values=[value for _,
                                     value, _ in node_intervention],
            )
        )
    return datasets


class CategoricalOCDDataModule(OCDDataModule):
    def __init__(
        self,
        # dataset (train and val)
        observation_size: int,
        intervention_size: int = 0,
        # configs to pass to bnlearn
        import_configs: th.Optional[th.Dict[str, th.Any]] = None,
        name: th.Optional[str] = None,
        # validation (split from train)
        val_size: th.Optional[th.Union[int, float]] = None,
        # seed
        seed: int = 0,
        # dataloader
        dl_args: th.Optional[th.Dict[str, th.Any]] = None,
        train_dl_args: th.Optional[th.Dict[str, th.Any]] = None,
        val_dl_args: th.Optional[th.Dict[str, th.Any]] = None,
        # batch_size
        batch_size: th.Optional[int] = 16,
        train_batch_size: th.Optional[int] = None,
        val_batch_size: th.Optional[int] = None,
        # pin memory
        pin_memory: bool = True,
        train_pin_memory: th.Optional[bool] = None,
        val_pin_memory: th.Optional[bool] = None,
        # shuffle
        train_shuffle: bool = True,
        val_shuffle: bool = False,
        # num_workers
        num_workers: int = 0,
        train_num_workers: th.Optional[int] = None,
        val_num_workers: th.Optional[int] = None,
        # extra parameters (ignored)
        **kwargs,
    ):
        """
        Datamodule for causal datasets.

        Args:
            name: name of the dataset (bnlearn dataset name or download url)
            observation_size: size of the observational data
            intervention_size: size of the interventional data (if 0, only observational data is used)
            import_configs: configs to pass to bnlearn.import_DAG
            val_size: size of the validation set (if 0 or None, no validation set is used)
            seed: seed for random_split
            dl_args: arguments to pass to DataLoader
            train_dl_args: arguments to pass to DataLoader for train set (overrides dl_args)
            val_dl_args: arguments to pass to DataLoader for val set (overrides dl_args)
            batch_size: batch size for train and val
            train_batch_size: batch size for train (overrides batch_size)
            val_batch_size: batch size for val (overrides batch_size)
            pin_memory: pin memory for train and val
            train_pin_memory: pin memory for train (overrides pin_memory)
            val_pin_memory: pin memory for val (overrides pin_memory)
            train_shuffle: shuffle for train
            val_shuffle: shuffle for val
            num_workers: num_workers for train and val
            train_num_workers: num_workers for train (overrides num_workers)
            val_num_workers: num_workers for val (overrides num_workers)
        """
        super().__init__(
            # dataset (train and val)
            observation_size=observation_size,
            # validation (split from train)
            val_size=val_size,
            # seed
            seed=seed,
            # dataloader
            dl_args=dl_args,
            train_dl_args=train_dl_args,
            val_dl_args=val_dl_args,
            # batch_size
            batch_size=batch_size,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            # pin memory
            pin_memory=pin_memory,
            train_pin_memory=train_pin_memory,
            val_pin_memory=val_pin_memory,
            # shuffle
            train_shuffle=train_shuffle,
            val_shuffle=val_shuffle,
            # num_workers
            num_workers=num_workers,
            train_num_workers=train_num_workers,
            val_num_workers=val_num_workers,
            # extra parameters (ignored)
            **kwargs,
        )
        self.intervention_size = intervention_size
        # set the import configs used for bnlearn
        self.import_configs = import_configs
        self.name = name

    def generate_datasets(self) -> th.List[BNLearnOCDDataset]:
        # override the generate dataset function with the one implemented for bnlearn here
        return generate_datasets(self.name,
                                 self.observation_size,
                                 self.intervention_size,
                                 show_progress=False,
                                 import_configs=self.import_configs)
