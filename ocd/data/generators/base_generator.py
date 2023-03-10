import typing as th
import numpy as np


class BaseGenerator:
    def __init__(self, seed: th.Optional[int] = None) -> None:
        # if seed is set to None, then set self.seed to a random number
        self.seed = seed or np.random.randint(0, 2 ** 32 - 1)
        self.sample_count = 0

    def get_iterative_seed(self):
        """
        Returns:
            int: a new seed based on the current seed and sample count and increases the sample count
        """
        self.sample_count += 1
        return self.seed + self.sample_count
