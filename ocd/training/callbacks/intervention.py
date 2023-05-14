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
        target: int = -1,
        percentile: float = 0.9,
        limit_y: float = 0.9, # 0 means no limit
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.flow_path = flow_path
        self.bins = bins
        self.num_samples = num_samples
        self.num_interventions = num_interventions
        self.k = k
        self.limit_y = limit_y
        self.target = target
        self.percentile = percentile
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
            model_samples = self.flow.do(idx=0, values=values, target=self.target, num_samples=self.num_samples).detach().cpu()
            model_means = model_samples.mean(-1)
            model_stds = model_samples.std(-1)
            data_samples = self.data.do(idx=0, values=values, target=self.target, num_samples=self.num_samples).detach().cpu()
            data_means = data_samples.mean(-1)
            data_stds = data_samples.std(-1)
            fig = plt.figure(figsize=(10, 10))
            # plot ate vs values of do with std as confidence interval
            plt.xlim(-self.k -1, self.k + 1)
            plt.xlabel("$do(X_1)$")
            tar = (self.data.data.shape[-1] + self.target) if (self.target < 0) else self.target
            plt.plot(values, data_means, label="Data", color="blue")
            plt.fill_between(
                values, (data_means - 3 * data_stds), (data_means + 3 * data_stds), alpha=0.2, color="blue"
            )
            original_limits = plt.ylim()
            if self.limit_y > 0:
                plt.ylim(abs(original_limits[0]) * (-1 - self.limit_y), abs(original_limits[1]) * (1 + self.limit_y))

            plt.ylabel("$\mathbb{E}_{X_1\sim Uniform(" + str(-self.k) + "," + str(self.k) + ")}[X_{" + str(tar + 1) + "} | do(X_1)]$")
            plt.plot(values, model_means, label="Model", color="red")
            plt.fill_between(
                values, (model_means - 3 * model_stds), (model_means + 3 * model_stds), alpha=0.2, color="red"
            )
            plt.legend()
            plt.grid()
            # based on self.data.base_distribution compute the 90% confidence interval for values
            # plot the confidence intervals as two vertical lines
            if self.percentile > 0:
                cis = [(1 - self.percentile)/2, 1 - (1 - self.percentile)/2]
                icis = self.data.base_distribution.icdf(torch.tensor(cis)).detach().cpu()
                plt.axvline(icis[0].item(), color="gray", linestyle="--")
                plt.axvline(icis[1].item(), color="gray", linestyle="--")
                # label the confidence interval
                plt.text(icis[0].item() + 0.1, plt.ylim()[1] -0.5, f"$X_1$ ${cis[0]*100: 0.1f}$% CI", rotation=90, verticalalignment="center", fontdict={"fontsize": 8})
                plt.text(icis[1].item() + 0.1, plt.ylim()[1] -0.5, f"$X_1$ ${cis[1]*100: 0.1f}$% CI", rotation=90, verticalalignment="center", fontdict={"fontsize": 8})

            plt.title(f"$N={self.data.data.shape[-1]}$")
            
            # convert the figure to a numpy array to be logged
            canvas = FigureCanvas(fig)
            canvas.draw()       # draw the canvas, cache the renderer

            image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
            plot = image_flat.reshape(*fig.canvas.get_width_height(), 3)  # (H, W, 3)

        trainer.logger.log_image("Data Fit", images=[plot], caption=["intervention"])
        return return_value
