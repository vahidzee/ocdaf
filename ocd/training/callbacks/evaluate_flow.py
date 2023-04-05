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
        reject_outliers_factor: float = 10,
        batch_size: int = 32,
    ):
        super().__init__(
            evaluate_every_n_epochs, evaluate_every_n_epoch_logic, epoch_buffer_size, log_training, log_validation
        )
        self.generator_batch_size = generator_batch_size
        self.reject_outliers_factor = reject_outliers_factor
        self.batch_size = batch_size

    def evaluate(self, trainer: pl.Trainer, pl_module: TrainingModule) -> None:
        all_inputs = self.all_logged_values["inputs"]

        if self.all_logged_values["perm_mat"][0] is not None:
            all_permutations = self.all_logged_values["perm_mat"]

        all_sampled = []
        for input, perm_mat in zip(all_inputs, all_permutations):
            # For each batch of the input data, decompose into multiple batches again
            # this is for some cuda out of memory issues
            for i in range(0, input.shape[0], self.batch_size):
                b = min(self.batch_size, input.shape[0] - i)
                sampled_values = pl_module.model.flow.sample(num_samples=b, perm_mat=perm_mat[i : i + b])
                all_sampled.append(sampled_values.detach().cpu())

        all_sampled = torch.cat(all_sampled, dim=0)
        all_inputs = torch.cat(all_inputs, dim=0)

        imgs = qqplot(
            all_inputs.detach().cpu(),
            all_sampled,
            self.reject_outliers_factor,
            "inputs",
            "sampled",
            image_size=(15, 10),
        )

        trainer.logger.log_image("qqplot", imgs, self.epoch_counter)
