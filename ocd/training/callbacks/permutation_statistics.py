from .logging import LoggingCallback
import typing as th
from lightning_toolbox import TrainingModule
import lightning.pytorch as pl
import torch
from ocd.models.permutation.utils import evaluate_permutations


class PermutationStatisticsCallback(LoggingCallback):
    """
    This callback generates statistics about the permutations that
    the permutation learning algorithm has generated. It is intended
    for taking a closer look at how the permutations look like and
    are generated. To see if there are a lot of soft permutations,
    how doubly stochastic they are, etc.
    """

    def __init__(
        self,
        evaluate_every_n_epochs: int = 1,
        # TODO: Change this into a method with dynamize
        evaluate_every_n_epoch_logic: th.Optional[str] = None,
        epoch_buffer_size: int = 1,
        log_training: bool = True,
        log_validation: bool = False,
        threshold: th.Optional[float] = None,
    ):
        super().__init__(
            evaluate_every_n_epoch_logic=evaluate_every_n_epoch_logic,
            evaluate_every_n_epochs=evaluate_every_n_epochs,
            epoch_buffer_size=epoch_buffer_size,
            log_training=log_training,
            log_validation=log_validation,
        )
        self.threshold = threshold

    def evaluate(self, trainer: pl.Trainer, pl_module: TrainingModule) -> None:
        logged_permutations = torch.cat(self.all_logged_values["all_perms_no_hard"], dim=0).detach().cpu()
        res = evaluate_permutations(logged_permutations, threshold=self.threshold)
        res = {f"explorability/{k}": v for k, v in res.items()}
        # log everything obtained in res which is a dictonary in pl_module.log
        pl_module.log_dict(res, on_step=False, on_epoch=True)
