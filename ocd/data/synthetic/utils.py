import numpy as np
from typing import Optional, Tuple, Union
from functools import partial

class RandomGenerator:  # TODO move it to utils + documentations
    def __init__(self, noise_type: str, seed: Optional[int], *args, **kwargs):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        if not hasattr(rng, noise_type):
            raise ValueError(f"Unknown noise type {noise_type}")

        self.rng = partial(getattr(rng, noise_type), *args, **kwargs)

    def __call__(self, size: Union[int, Tuple[int, ...]]):
        return self.rng(size=size)