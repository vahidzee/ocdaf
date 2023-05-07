import typing as th
import abc
import networkx as nx
import torch
from evaluation import backward_relative_penalty, count_backward, count_SHD, count_SID
import dypy


class AbstractBaseline(abc.ABC):
    """Abstract class for running baselines that estimate causal orderings.

    Given a OCDDataset object (which has a dag and samples attribute), the baseline
    should estimate the causal ordering of the variables in the dataset.
    """

    def __init__(
            self,
            dataset: th.Union["OCDDataset", str],  # type: ignore
            name: th.Optional[str] = None,
            dataset_args: th.Optional[th.Dict[str, th.Any]] = None,
    ):
        self.name = self.__class__.__name__ if name is None else name
        dataset = dypy.get_value(dataset) if isinstance(dataset, str) else dataset
        dataset_args = dict() if dataset_args is None else dataset_args
        self.dataset = dataset if isinstance(dataset, torch.utils.data.Dataset) else dataset(**dataset_args)

    @property
    def true_ordering(self) -> th.List[int]:
        """Return the true ordering of the dataset."""
        if self.dataset is None or not hasattr(self.dataset, "dag"):
            raise ValueError("Dataset must have a dag attribute to return the true ordering.")
        return list(nx.topological_sort(self.dataset.dag))

    def get_data(self, conversion: th.Literal["tensor", "numpy", "pandas"] = "tensor"):
        if self.dataset is None or not hasattr(self.dataset, "dag"):
            raise ValueError("Dataset is not loaded to get the samples.")
        if conversion == "tensor":
            return torch.from_numpy(self.dataset.samples.to_numpy())
        elif conversion == "numpy":
            return self.dataset.samples.to_numpy()
        return self.dataset.samples

    @abc.abstractmethod
    def estimate_order(self, **kwargs) -> th.Union[th.List[int], torch.Tensor]:
        """Fit the baseline on the dataset and return the estimated causal orderings.
        Args:
            dataset: The dataset to fit on

        Returns:
            A list of the estimated orderings, e.g., [2, 0, 1] means X_2 -> X_0 -> X_1
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def estimate_dag(self, **kwargs) -> th.Union[nx.DiGraph, torch.Tensor]:
        raise NotImplementedError()

    def evaluate(self, structure: bool = False):
        """Evaluate the baseline on the dataset.
        Args:
            structure: Whether to evaluate the structure of the estimated DAG

        Returns:
            A dictionary of evaluation metrics
        """
        estimated_order = self.estimate_order()
        # count the number of backward edges
        backward_count = count_backward(estimated_order, self.dataset.dag)
        # compute the backward relative penalty
        backward_penalty = backward_relative_penalty(estimated_order, self.dataset.dag)
        result = {
            "backward_count": backward_count,
            "backward_relative_penalty": backward_penalty,
            "true_ordering": self.true_ordering,
            "estimated_ordering": estimated_order,
        }
        if structure:
            estimated_dag = self.estimate_dag()
            result['SID'] = count_SID(self.dataset.dag, estimated_dag)
            result['SHD'] = count_SHD(self.dataset.dag, estimated_dag)

        return result
