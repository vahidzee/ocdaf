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

    def __init__(self, image_size: th.Optional[th.Tuple[int, int]] = None,
                 show_functions_in_dag: th.Optional[bool] = True,
                 show_original_statistics_on_histograms: th.Optional[bool] = True):
        super().__init__()
        self.image_size = image_size
        self.show_functions_in_dag = show_functions_in_dag
        self.show_original_statistics_on_histograms = show_original_statistics_on_histograms

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

        caption_for_histogram = "Histograms of the data"
        fig, ax = plt.subplots()
        # customize the image size if needed
        if self.image_size:
            fig.set_size_inches(self.image_size[0], self.image_size[1])
        try:
            dataset = trainer.datamodule.data
            for c in dataset.samples.columns:
                mean = dataset.samples_statistics[c]['mean']
                std = dataset.samples_statistics[c]['std']
                label = f"x_{c} : "
                caption_for_histogram += f"\nOriginal mean and std for x_{c}: {mean:.2f} +- {std:.2f}"
                ax.hist(dataset.samples[c], density=True, bins=100, alpha=0.5, label=label)

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
        
        caption_for_graph = trainer.datamodule.data.explanation if self.show_functions_in_dag else "Causal Graph"
        caption_for_histogram = caption_for_histogram if self.show_original_statistics_on_histograms else "Histograms of the data"
        trainer.logger.log_image("Data", images=imgs, caption=[caption_for_graph, caption_for_histogram])

        return super().on_fit_start(trainer, pl_module)
