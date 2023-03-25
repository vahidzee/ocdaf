from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningModule, LightningDataModule
from lightning_toolbox import TrainingModule
from lightning_toolbox import DataModule
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def softplus(x):
    return np.log(1 + np.exp(x))


def main():
    cli = LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        # auto_configure_optimizers=False,
        run=False,
    )
    cli.datamodule.setup("fit")
    # cli = LightningCLI(
    #     LightningModule,
    #     LightningDataModule,
    #     # run=False,
    # )
    # cli.trainer.fit(cli.datamodule)
    nx.draw(cli.datamodule.data.dag, with_labels=True)
    plt.show()
    print(cli.datamodule.data.explanation)
    plt.hist(cli.datamodule.data.samples[0], density=True, bins=100, alpha=0.5, label="x0")
    plt.hist(cli.datamodule.data.samples[1], density=True, bins=100, alpha=0.5, label="x1")
    plt.hist(cli.datamodule.data.samples[2], density=True, bins=100, alpha=0.5, label="x2")
    plt.legend()
    plt.show()

    # x2 = softplus(0.80) + softplus(0.84) * np.random.normal(0, 1, 100000)
    # x0 = softplus(x2 * 0.54 + 0.57) + softplus(x2 * 0.67 + 0.61) * np.random.normal(0, 1, 100000)
    # x1 = softplus(x2 * 0.62 + x0 * 0.81 + 0.58) + softplus(x2 * 0.52 + x0 * 0.75 + 0.88) * np.random.normal(
    #     0, 1, 100000
    # )
    # plt.hist(x0, density=True, bins=100, alpha=0.5, label="x0")
    # plt.hist(x1, density=True, bins=100, alpha=0.5, label="x1")
    # plt.hist(x2, density=True, bins=100, alpha=0.5, label="x2")
    # plt.legend()
    # plt.show()


# x(2) = softplus(+0.80) + softplus(+0.84) * N(0.00,1.00)
# ---------------------
# x(0) = softplus(x(2)*0.54+0.57) + softplus(x(2)*0.67+0.61) * N(0.00,1.00)
# ---------------------
# x(1) = softplus(x(2)*0.62+x(0)*0.81+0.58) + softplus(x(2)*0.52+x(0)*0.75+0.88) * N(0.00,1.00)
# ---------------------

if __name__ == "__main__":
    main()
