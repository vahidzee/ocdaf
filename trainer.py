from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningModule, LightningDataModule


def main():
    cli = LightningCLI(LightningModule, LightningDataModule, subclass_mode_model=True, subclass_mode_data=True)


if __name__ == "__main__":
    main()
