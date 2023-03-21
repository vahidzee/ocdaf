from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningModule, LightningDataModule
from lightning_toolbox import TrainingModule


def main():
    cli = LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    main()
