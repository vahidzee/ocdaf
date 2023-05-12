from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning_toolbox import TrainingModule


# TODO: Make sure to submit an issue on Pytorch Lightning about the incompatibility of manual optimization with multiple
# optimizers and also using ModelCheckpoint callback. The bug that we found was that global_step was not being updated
# and it remained equal to zero.
class DebuggedModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_pl_module_phase = None
        self.edge_detected = False

    def on_train_epoch_end(self, trainer: Trainer, pl_module: TrainingModule) -> None:
        ret = super().on_train_epoch_end(trainer, pl_module)
        if hasattr(pl_module, "current_phase") and self.last_pl_module_phase != pl_module.current_phase:
            self.last_pl_module_phase = pl_module.current_phase
            self.edge_detected = True
        else:
            self.edge_detected = False
        return ret

    def _should_skip_saving_checkpoint(self, trainer: pl.Trainer) -> bool:
        from lightning.pytorch.trainer.states import TrainerFn

        # if edge is not detected, then skip
        if not self.edge_detected:
            return True

        # ignore the condition on global_step
        return (
            bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
        )
