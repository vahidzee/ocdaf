"""
This callback contains some functionalities to evaluate how good the flow performs
"""

# from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import Callback
import lightning.pytorch as pl
from lightning_toolbox import TrainingModule
import torch
from ocd.visualization.qqplot import qqplot
import typing as th
import os
import yaml
from lightning.pytorch import Trainer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from lightning.pytorch.loggers import WandbLogger


class DataVisualizer(Callback):
    """
    This callback simply visualizes the data that is being used for training
    """

    def __init__(self, image_size: th.Optional[th.Tuple[int, int]] = None):
        super().__init__()
        self.image_size = image_size

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        imgs = []

        fig, ax = plt.subplots()
        # customize the image size if needed
        if self.image_size:
            fig.set_size_inches(self.image_size[0], self.image_size[1])
        try:
            nx.draw(trainer.datamodule.data.dag, with_labels=True, ax=ax)

            ax.set_title("Graph of DGM")

            # draw everything to the figure for conversion
            fig.canvas.draw()
            # convert the figure to a numpy array
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            imgs.append(data)
        finally:
            plt.close()

        fig, ax = plt.subplots()
        # customize the image size if needed
        if self.image_size:
            fig.set_size_inches(self.image_size[0], self.image_size[1])
        try:
            for c in trainer.datamodule.data.samples.columns:
                ax.hist(trainer.datamodule.data.samples[c], density=True, bins=100, alpha=0.5, label=f"x_{c}")

            ax.legend()
            ax.set_title("Histograms of the data")

            # draw everything to the figure for conversion
            fig.canvas.draw()
            # convert the figure to a numpy array
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            imgs.append(data)
        finally:
            plt.close()

        trainer.logger.log_image("Data", images=imgs, caption=[trainer.datamodule.data.explanation, "Histograms"])

        return super().on_fit_start(trainer, pl_module)
