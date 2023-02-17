from lightning.pytorch.cli import LightningCLI
from ocd.training import OrderedTrainingModule
from ocd.data import OCDDataModule


def main():
    LightningCLI(OCDDataModule, OrderedTrainingModule)


if __name__ == "__main__":
    main()
