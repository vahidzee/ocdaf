"""
This file contains the synthetic datasets generated.
"""
from .base_dataset import OCDDataset
import typing as th
from .scm import SCMGenerator
import dypy
from lightning_toolbox import DataModule


class SyntheticOCDDataset(OCDDataset):
    def __init__(
        self,
        observation_size: int,
        scm_generator: th.Union[SCMGenerator, str],
        scm_generator_args: th.Optional[th.Dict[str, th.Any]] = None,
        seed: th.Optional[int] = None,
        name: th.Optional[str] = None,
        enable_simulate: bool = True,
    ):
        """
        Args:
            scm_generator: an SCM Generator object or a string which contains the class directory
                that is a Pytorch Dataset object used.
            scm_generator_args: arguments to pass to the SCM Generator if it is a string
            seed: The seed which is used to simulate the data
            name: The name of the dataset
        """
        scm_generator = (
            scm_generator
            if isinstance(scm_generator, SCMGenerator)
            else dypy.eval(scm_generator)(**scm_generator_args)
        )
        self.seed = seed
        self.scm = scm_generator.generate_scm()
        if enable_simulate:
            df = self.scm.simulate(observation_size, seed=self.seed)
        else:
            df = None
        super().__init__(samples=df, dag=self.scm.dag, name=name, explanation=self.scm.get_description())
