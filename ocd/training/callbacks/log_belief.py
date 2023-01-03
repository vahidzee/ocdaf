import typing as th
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl


class BeliefCallback(Callback):
    def __init__(
        self,
        permutation_log_frequency: int = 1,
        visualize_permutations: bool = True,
        visualize_mask_guiders: bool = True,
        n_sample_mask_guiders: int = 4,
    ):
        self.permutation_log_frequency = permutation_log_frequency
        self.current_permutation = 0
        self.permutation_history = None
        self.visualize_permutations = visualize_permutations
        self.visualize_mask_guiders = visualize_mask_guiders
        self.n_sample_mask_guiders = n_sample_mask_guiders

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if pl_module.get_phase() == "maximization":
            # Short circuit if we are in the maximization phase
            # Because we are not updating the permutation matrix belief system at all
            return

        new_value = (self.current_permutation + 1) % self.permutation_log_frequency

        if self.current_permutation == 0:
            # don't log in this case
            self.current_permutation = new_value
            return
        self.current_permutation = new_value

        # get the root tensorboard logger
        logger = pl_module.logger.experiment

        if self.visualize_permutations:
            visualize_permutations(pl_module, self, logger)

        if self.visualize_mask_guiders:
            visualize_mask_guiders(pl_module, self, logger)

        return super().on_train_epoch_end(trainer, pl_module)


def visualize_permutations(
    pl_module: pl.LightningModule,
    permutation_callback: BeliefCallback,
    logger: pl.loggers.TensorBoardLogger,
):
    """
    This function visualizes the permutation matrix and saves it to the tensorboard logger.
    For visualization, it takes a hard permutation sample from the latent permutation model
    and it plots the history of all the permutations up until the current point.

    Args:
        pl_module: The training module itself
        permutation_callback: the callback
        logger: the tensorboard logger
    Returns:
        None
    """
    permutation = permutation_callback.model.latent_permutation_model.hard_permutation(
        return_matrix=False, training_module=pl_module
    )

    # plot the permutation using matplotlib and save it to a numpy array
    fig, ax = plt.subplots()
    try:
        # concatenate permutation to the history as a new column
        if permutation_callback.permutation_history is None:
            permutation_callback.permutation_history = np.array([permutation])
        else:
            permutation_callback.permutation_history = np.concatenate(
                (permutation_callback.permutation_history, np.array([permutation]))
            )

        # print(self.permutation_history.shape)
        for i in range(permutation_callback.permutation_history.shape[1]):
            # print(self.permutation_history[:, i])
            ax.plot(permutation_callback.permutation_history[:, i], label=f"p_{i}", alpha=0.5)

        ax.legend()
        ax.set_title(f"Permutation")
        ax.set_xlabel("epoch")
        ax.set_ylabel("permutation")
        fig.canvas.draw()
        # convert the figure to a numpy array
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # log the figure to tensorboard
        logger.add_image(f"belief_system/permutation", data, pl_module.current_epoch, dataformats="HWC")
    finally:
        plt.close()


def visualize_mask_guiders(
    pl_module: pl.LightningModule, permutation_callback: BeliefCallback, logger: pl.loggers.TensorBoardLogger
):
    """
    This function is used to visualize the current belief system.
    It will first start off by sampling a bunch of permutation matrices according to the current belief system.
    Then it will plot the sampled matrices as a heatmap. Furthermore, for better checking, it will also plot the row sum
    and column of the sampled matrices as a histogram. This will help in determining whether the end matrix is a permutation
    matrix or not.

    By the end, it will also plot the current belief system itself as a heatmap as well. Since the belief system is an
    n by n matrix, it will be plotted as a heatmap.

    Args:
        pl_module: the lightning module in charge of the training
        permutation_callback: the callback
        logger: the tensorboard logger
    Returns:
        None
    """
    fig, ax = plt.subplots()
    try:
        mask_guiders = pl_module.model.latent_permutation_model.soft_permutation(
            num_samples=permutation_callback.n_sample_mask_guiders, training_module=pl_module
        )

        for i, mask_guider in enumerate(mask_guiders):
            # plot self.mask_guider as a heatmap
            ax.imshow(mask_guider, interpolation="none")
            ax.set_title(f"mask_guider (represent a permanent matrix)")
            sm0 = mask_guider.sum(axis=0)
            sm1 = mask_guider.sum(axis=1)

            ax.set_xlabel(f"row sum in [{round(sm0.min().item(), 2)}, {round(sm0.max().item(), 2)}]")
            ax.set_ylabel(f"row sum in [{round(sm1.min().item(), 2)}, {round(sm1.max().item(), 2)}]")

            fig.canvas.draw()
            # convert the figure to a numpy array
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # log the figure to tensorboard
            logger.add_image(f"belief_system/mask_guider_{i}", data, pl_module.current_epoch, dataformats="HWC")
    finally:
        plt.close()

    fig, ax = plt.subplots()
    try:
        belief = pl_module.model.latent_permutation_model.get_belief_system()
        # plot self.mask_guider as a heatmap
        ax.imshow(belief, interpolation="none")
        ax.set_title(f"Gamma (summarizes the belief parameters)")
        ax.set_xlabel(f"all_values in [{round(belief.min().item(), 2)}, {round(belief.max().item(), 2)}")
        fig.canvas.draw()
        # convert the figure to a numpy array
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # log the figure to tensorboard
        logger.add_image(f"belief_system/gamma", data, pl_module.current_epoch, dataformats="HWC")
    finally:
        plt.close()
