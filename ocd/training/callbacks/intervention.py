import typing as th
from lightning_toolbox import TrainingModule
from matplotlib import pyplot as plt
import lightning.pytorch as pl
import torch
import numpy as np
import dypy as dy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class InterventionCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs: int = 1,
        flow_path: str = "model.flow",
        bins: int = 100,
        num_samples: int = 200,
        num_interventions: int = 50,
        k: int = 5,
        sample: bool = True,
        intervene: bool = True,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.flow_path = flow_path
        self.bins = bins
        self.num_samples = num_samples
        self.num_interventions = num_interventions
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
            values = torch.linspace(-self.k, self.k, self.num_interventions)
            model_samples = self.flow.do(idx=0, values=values, target=-1, num_samples=self.num_samples).detach().cpu()
            model_means = model_samples.mean(-1)
            model_stds = model_samples.std(-1)
            data_samples = self.data.do(idx=0, values=values, target=-1, num_samples=self.num_samples).detach().cpu()
            data_means = data_samples.mean(-1)
            data_stds = data_samples.std(-1)
            fig = plt.figure(figsize=(10, 10))
            # plot ate vs values of do with std as confidence interval
            plt.xlim(-self.k -1, self.k + 1)
            plt.plot(values, model_means, label="Model", color="red")
            plt.fill_between(
                values, (model_means - 3 * model_stds), (model_means + 3 * model_stds), alpha=0.2, color="red"
            )
            plt.plot(values, data_means, label="Data", color="blue")
            plt.fill_between(
                values, (data_means - 3 * data_stds), (data_means + 3 * data_stds), alpha=0.2, color="blue"
            )

            canvas = FigureCanvas(fig)
            canvas.draw()       # draw the canvas, cache the renderer

            image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
            plot = image_flat.reshape(*fig.canvas.get_width_height(), 3)  # (H, W, 3)

        trainer.logger.log_image("Data Fit", images=[plot], caption=["intervention"])
        return return_value
