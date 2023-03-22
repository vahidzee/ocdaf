"""
This callback contains some functionalities to evaluate how good the flow performs
"""

# from lightning.pytorch.callbacks import Callback
from .logging import LoggingCallback
import lightning.pytorch as pl
from lightning_toolbox import TrainingModule
import torch
from ocd.visualization.qqplot import qqplot
import typing as th


class EvaluateFlow(LoggingCallback):
    def __init__(
        self,
        generator_batch_size: int = 100,
        evaluate_every_n_epochs: int = 1,
        evaluate_every_n_epoch_logic: th.Optional[str] = None,
        epoch_buffer_size: int = 1,
        log_training: bool = True,
        log_validation: bool = False,
    ):
        super().__init__(
            evaluate_every_n_epochs, evaluate_every_n_epoch_logic, epoch_buffer_size, log_training, log_validation
        )
        self.generator_batch_size = generator_batch_size

    def evaluate(self, trainer: pl.Trainer, pl_module: TrainingModule) -> None:
        all_inputs = torch.cat(self.all_logged_values["inputs"], dim=0)
        all_permutations = torch.cat(self.all_logged_values["perm_mat"], dim=0)

        N = all_inputs.shape[0]
        B = min(self.generator_batch_size, N)
        all_sampled = []
        for i in range(0, N, B):
            b = min(B, N - i)
            perm_mats = all_permutations[i : i + b]
            sampled_values = pl_module.model.flow.sample(num_samples=b, perm_mat=perm_mats)
            all_sampled.append(sampled_values)

        all_sampled = torch.cat(all_sampled, dim=0)

        imgs = qqplot(all_inputs, all_sampled, "inputs", "sampled", image_size=(15, 10))

        trainer.logger.log_image("qqplot", imgs, self.epoch_counter)
