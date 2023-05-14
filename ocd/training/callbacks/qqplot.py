import typing as th
from lightning_toolbox import TrainingModule
import lightning.pytorch as pl
from matplotlib import pyplot as plt
import torch
import numpy as np
import dypy as dy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def qqplot(x: torch.Tensor, y: torch.Tensor, bins=200, alpha=0.5, x_label="x", y_label="y"):
    # x and y are tensors of shape (N, D)
    # we want to plot the qqplot for i==j to get a grid of (1 x D) plots
    # assert x.shape == y.shape # number of samples could be different
    d = x.shape[-1]
    fig, axs = plt.subplots(d, figsize=(20, 20))
    for i in range(d):
        axs[i].hist(x[:, i].flatten(), bins=bins, alpha=alpha, label=x_label, density=True, color="blue")
        axs[i].hist(y[:, i].flatten(), bins=bins, alpha=alpha, label=y_label, density=True, color="red")
        axs[i].legend()
    return fig


class QQplotCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs: int = 5,
        flow_path: str = "model.flow",
        bins: int = 100,
        num_samples: int = 1000,
        k: int = 5,
        sample: bool = True,
        intervene: bool = True,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.flow_path = flow_path
        self.bins = bins
        self.num_samples = num_samples
        self.sample = sample
        self.intervene = intervene
        self.k = k
        self.flow = None  # read it on fit
        self.data = None  # read it on fit

    def on_fit_start(self, trainer: pl.Trainer, pl_module: TrainingModule) -> None:
        self.flow = dy.get_value(self.flow_path, context=pl_module)
        self.data = trainer.datamodule.data  # it only supports InterventionChainData
        return super().on_fit_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:  # type: ignore
        return_value = super().on_train_epoch_end(trainer, pl_module)
        if self.every_n_epochs is not None and self.every_n_epochs > 0:
            if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
                return return_value

        with torch.no_grad():
            model_samples = self.flow.sample(self.num_samples).detach().cpu()
            data_samples = self.data.data.detach().cpu()
            fig = qqplot(data_samples, model_samples, bins=self.bins, x_label="data", y_label="model")

            canvas = FigureCanvas(fig)
            canvas.draw()       # draw the canvas, cache the renderer

            image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
            plot = image_flat.reshape(*fig.canvas.get_width_height(), 3)  # (H, W, 3)


        trainer.logger.log_image("Data Fit", images=[plot], caption=["qqplot"])
        return return_value
