import typing as th
from lightning_toolbox import TrainingModule
from matplotlib import pyplot as plt
import lightning.pytorch as pl
import torch
import numpy as np
import dypy as dy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

plt.rcParams.update({"font.size": 18})


def _draw_ax(
    ax,
    n,
    k,
    values=None,
    pred_means=None,
    pred_stds=None,
    gt_means=None,
    gt_stds=None,
    target=-1,
    percentile=0.95,
    limit_y=0.9,
    limit_ys=None,
    icis=None,
):
    if pred_means is not None:
        pred_mean = pred_means[:, target]
        pred_std = pred_stds[:, target]
    if gt_means is not None:
        gt_mean = gt_means[:, target]
        gt_std = gt_stds[:, target]
    ax.set_xlim(-k, k)
    ax.set_xlabel("$do(X_1)$")
    tar = (n + target) if (target < 0) else target
    if gt_means is not None:
        ax.plot(values, gt_mean, label="True Density", color="blue")
        ax.fill_between(values, (gt_mean - 3 * gt_std), (gt_mean + 3 * gt_std), alpha=0.2, color="blue")
    original_limits = ax.get_ylim()
    if limit_ys is not None:
        ax.set_ylim(limit_ys[0], limit_ys[1])
    elif limit_y > 0:
        ax.set_ylim(abs(original_limits[0]) * (-1 - limit_y), abs(original_limits[1]) * (1 + limit_y))

    ax.set_ylabel("$P(X_{" + str(tar + 1) + "} | do(X_1))$")
    if pred_means is not None:
        ax.plot(values, pred_mean, label="Predicted Density", color="red")
        ax.fill_between(values, (pred_mean - 3 * pred_std), (pred_mean + 3 * pred_std), alpha=0.2, color="red")
    ax.legend(loc="lower right")
    # plot the confidence intervals as two vertical lines
    if percentile > 0:
        ax.axvline(icis[0].item(), color="gray", linestyle="--")
        ax.axvline(icis[1].item(), color="gray", linestyle="--")
        # draw a horizontal line with two arrows pointing to the vertical lines
        ax.annotate(
            "",
            xy=(icis[0].item(), ax.get_ylim()[1] - 1.25),
            xytext=(icis[1].item(), ax.get_ylim()[1] - 1.25),
            arrowprops=dict(arrowstyle="<->", color="gray"),
        )
        # label the confidence interval
        ax.text(
            (icis[0].item() + icis[1].item()) / 2,
            ax.get_ylim()[1] - 1.75,
            f"${percentile*100: 0.1f}$% CI for $X_1$",
            horizontalalignment="center",
            verticalalignment="center",
            fontdict={"fontsize": 15},
        )


def draw_grid(
    n,
    k,
    target=None,
    values=None,
    pred_means=None,
    pred_stds=None,
    gt_means=None,
    gt_stds=None,
    percentile=0.95,
    limit_y=0.9,
    limit_ys=None,
    icis=None,
    fignaxes=None,
):
    targets = target if target is not None else list(range(n))
    if fignaxes is None:
        fig, axs = plt.subplots(1, len(targets), figsize=(8 * len(targets), 8))  # Adjust the figure size as necessary
    else:
        fig, axs = fignaxes

    for i, target in enumerate(targets):
        _draw_ax(
            ax=axs[i],
            n=n,
            k=k,
            values=values,
            pred_means=pred_means,
            pred_stds=pred_stds,
            gt_means=gt_means,
            gt_stds=gt_stds,
            target=target,
            percentile=percentile,
            limit_y=limit_y,
            limit_ys=limit_ys,
            icis=icis,
        )

    return fig, axs


def draw(
    fig,
    n,
    k,
    values=None,
    pred_means=None,
    pred_stds=None,
    gt_means=None,
    gt_stds=None,
    target=-1,
    percentile=0.95,
    limit_y=0.9,
    limit_ys=None,
    icis=None,
):
    if pred_means is not None:
        pred_mean = pred_means[:, target]
        pred_std = pred_stds[:, target]
    if gt_means is not None:
        gt_mean = gt_means[:, target]
        gt_std = gt_stds[:, target]
    plt.xlim(-k, k)
    plt.xlabel("$do(X_1)$")
    tar = (n + target) if (target < 0) else target
    if gt_means is not None:
        plt.plot(values, gt_mean, label="True Density", color="blue")
        plt.fill_between(values, (gt_mean - 3 * gt_std), (gt_mean + 3 * gt_std), alpha=0.2, color="blue")
    original_limits = plt.ylim()
    if limit_ys is not None:
        plt.ylim(limit_ys[0], limit_ys[1])
    elif limit_y > 0:
        plt.ylim(abs(original_limits[0]) * (-1 - limit_y), abs(original_limits[1]) * (1 + limit_y))

    plt.ylabel("$P(X_{" + str(tar + 1) + "} | do(X_1))$")
    if pred_means is not None:
        plt.plot(values, pred_mean, label="Predicted Density", color="red")
        plt.fill_between(values, (pred_mean - 3 * pred_std), (pred_mean + 3 * pred_std), alpha=0.2, color="red")
    plt.legend(loc="lower right")
    # plt.grid()
    # based on data.base_distribution compute the 90% confidence interval for values
    # plot the confidence intervals as two vertical lines
    if percentile > 0:
        plt.axvline(icis[0].item(), color="gray", linestyle="--")
        plt.axvline(icis[1].item(), color="gray", linestyle="--")
        # label the confidence interval
        # create a horizontal line with two arrows pointing to the vertical lines
        plt.annotate(
            "",
            xy=(icis[0].item(), plt.ylim()[1] - 1.25),
            xytext=(icis[1].item(), plt.ylim()[1] - 1.25),
            arrowprops=dict(arrowstyle="<->", color="gray"),
        )
        # label the confidence interval
        plt.text(
            (icis[0].item() + icis[1].item()) / 2,
            plt.ylim()[1] - 1.75,
            f"${percentile*100: 0.1f}$% CI for $X_1$",
            horizontalalignment="center",
            verticalalignment="center",
            fontdict={"fontsize": 15},
        )
    return fig

    # set x ticks to have values at every 2.5 from zero to both sides
    # plt.xticks((-torch.arange(0, k, 2.5)).tolist()[1:][::-1] + torch.arange(0, k, 2.5).tolist())


class InterventionCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs: int = 1,
        flow_path: str = "model.flow",
        num_samples: int = 100,
        num_interventions: int = 250,
        k: float = 5,
        target: th.Union[int, th.List[int], None] = None,
        percentile: float = 0.9,
        limit_y: float = 0.9,  # 0 means no limit
        limit_ys: th.Optional[th.List[float]] = None,
        title: str = "Interventions",
        caption: str = "Interventions",
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.flow_path = flow_path
        self.num_samples = num_samples
        self.num_interventions = num_interventions
        self.k = k
        self.limit_y = limit_y
        self.limit_ys = limit_ys
        self.target = target
        self.percentile = percentile
        self.title = title
        self.caption = caption
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

        # data shape
        n = self.data.data.shape[-1]

        # precentiles
        cis, icis = None, None
        if self.percentile:
            cis = [(1 - self.percentile) / 2, 1 - (1 - self.percentile) / 2]
            icis = self.data.base_distribution.icdf(torch.tensor(cis)).detach().cpu()

        # predictions
        with torch.no_grad():
            values = torch.linspace(-self.k, self.k, self.num_interventions)
            pred_samples = self.flow.do(idx=0, values=values, num_samples=self.num_samples).detach().cpu()
            pred_means = pred_samples.mean(-2)
            pred_stds = pred_samples.std(-2)
            gt_samples = self.data.do(idx=0, values=values, num_samples=self.num_samples).detach().cpu()
            data_means = gt_samples.mean(-2)
            data_stds = gt_samples.std(-2)

        if self.target is None or isinstance(self.target, list):
            targets = self.target if self.target is not None else list(range(n))
            nrows = len(targets)
            fig, axes = draw_grid(
                fignaxes=plt.subplots(1, len(targets), figsize=(8 * nrows, 8)),  # Adjust the figure size as necessary
                n=n,
                k=self.k,
                values=values,
                pred_means=pred_means,
                pred_stds=pred_stds,
                gt_means=data_means,
                gt_stds=data_stds,
                target=self.target,
                percentile=self.percentile,
                limit_y=self.limit_y,
                limit_ys=self.limit_ys,
                icis=icis,
            )
        else:
            fig = draw(
                fig=plt.figure(figsize=(8, 8)),
                n=n,
                k=self.k,
                values=values,
                pred_means=pred_means,
                pred_stds=pred_stds,
                gt_means=data_means,
                gt_stds=data_stds,
                target=self.target,
                percentile=self.percentile,
                limit_y=self.limit_y,
                limit_ys=self.limit_ys,
                icis=icis,
            )

        # convert the figure to a numpy array to be logged
        canvas = FigureCanvas(fig)
        canvas.draw()  # draw the canvas, cache the renderer

        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")  # (H * W * 3,)
        plot = image_flat.reshape(*fig.canvas.get_width_height(), 3)  # (H, W, 3)

        trainer.logger.log_image(self.title, images=[plot], caption=[self.caption])
        return return_value
