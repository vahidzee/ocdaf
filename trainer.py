from lightning.pytorch.cli import LightningCLI
from ocd.training import OrderedTrainingModule
from lightning_toolbox.data import DataModule


def main():
    cli = LightningCLI(OrderedTrainingModule,
                       DataModule)


if __name__ == "__main__":
    main()
