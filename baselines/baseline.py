from ocd.data import OCDDataset
import typing as th


class Baseline:
    def __init__(self, name: str):
        """ Args:
            name: The name of the baseline
        """
        self.name = name

    def fit(self, dataset: OCDDataset) -> th.List[int]:
        """ Fit the baseline on the dataset
        Args:
            dataset: The dataset to fit on

        Returns:
            A list of the estimated orderings, e.g., [2, 0, 1] means X_2 -> X_0 -> X_1
        """
        raise NotImplementedError()
