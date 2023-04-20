from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningModule, LightningDataModule
from lightning_toolbox import TrainingModule
import os

def main():
    cli = LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    # delete config.yaml file if it exists
    #
    # This is due to an exception in the LightningCLI
    # that occurs when the config.yaml file already exists
    if os.path.exists("config.yaml"):
        os.remove("config.yaml")
        
    main()
