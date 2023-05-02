from lightning_toolbox import TrainingModule
import typing as th
import torch


class OCDafTrainingModule(TrainingModule):
    def __init__(
        self,
        *args,
        maximization_specifics: th.Optional[dict] = None,
        expectation_specifics: th.Optional[dict] = None,
        grad_clip_val: th.Optional[float] = 1.0,
        phases: th.List[str] = ["maximization", "expectation"],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

        self.reset_optimizers()

        if grad_clip_val:
            for p in self.model.parameters():
                p.register_hook(lambda grad: grad.clamp(-grad_clip_val, grad_clip_val))
        self.phases = phases
        self.map_phase_to_idx = {phase: i for i, phase in enumerate(self.phases)}

        # for each scheduler have the running average
        self.running_avg = [0 for _ in self._schedulers] if self._schedulers is not None else None

        self.cnt = 0

    def reinitialize_flow_weights(self):
        def func(m):
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        self.model.flow.apply(func)

    def step(
        self,
        batch: th.Optional[th.Any] = None,
        batch_idx: th.Optional[int] = None,
        name: str = "train",
        return_results: bool = False,
        return_factors: bool = False,
        log_results: bool = True,
        **kwargs,  # additional arguments to pass to the objective
    ):
        """Train or evaluate the model with the given batch.
        Args:
            batch: batch of data to train or evaluate with
            batch_idx: index of the batch
            optimizer_idx: index of the optimizer
            name: name of the step ("train" or "val")
        Returns:
            None if the model is in evaluation mode, else a tensor with the training objective
        """

        if hasattr(self, "current_phase"):
            optimizer_idx = self.map_phase_to_idx[self.current_phase]
        else:
            optimizer_idx = 0

        is_val = name == "val"
        assert hasattr(self, "objective"), "No objective defined"

        if not is_val and not self.is_optimizer_active(
            optimizer_idx=optimizer_idx, batch_idx=batch_idx, epoch=self.current_epoch
        ):
            return None

        results, factors = self.objective(
            batch=batch,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
            training_module=self,
            return_factors=True,
            **kwargs,
        )
        if log_results:
            self.log_step_results(results, factors, name)
        if not is_val:
            opt = self._optimizers[optimizer_idx]
            opt.zero_grad()
            loss = results["loss"]
            self.manual_backward(loss)

            # Manually adding
            # self.trainer.global_step += 1

            opt.step()

            self.log(f"lr/{optimizer_idx}", opt.param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=True)

            return loss.mean() if isinstance(loss, torch.Tensor) else loss
        elif self._schedulers is not None:
            for i, sched in enumerate(self._schedulers):
                monitor_key = sched["monitor"]
                self.running_avg[i] = (
                    self.running_avg[i] * batch_idx + results[monitor_key].detach().cpu().item()
                ) / (batch_idx + 1)

    def on_train_epoch_end(self):
        ret = super().on_train_epoch_end()
        if self._schedulers is None:
            return ret
        for scheduler_idx in range(len(self._schedulers)):
            # TODO: This is hacky, but I don't know how to do it better
            if hasattr(self, "current_phase") and self.current_phase not in self._schedulers[scheduler_idx]["name"]:
                continue

            scheduler = self._schedulers[scheduler_idx]["scheduler"] if self._schedulers else None
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self.running_avg[scheduler_idx])
                else:
                    scheduler.step()
        return ret

    def training_step(self, batch, batch_idx, **kwargs):
        return self.step(batch, batch_idx, name="train", **kwargs)

    def _get_optimizers(self):
        return self._optimizers

    def reset_optimizers(self):
        self._optimizers = self.configure_optimizers()
        self._schedulers = None
        if isinstance(self._optimizers, tuple):
            self._optimizers, self._schedulers = self._optimizers
        elif (
            isinstance(self._optimizers, dict)
            and "optimizer" in self._optimizers
            and "lr_scheduler" in self._optimizers
        ):
            self._schedulers = self._optimizers.get("lr_scheduler", None)
            if self._schedulers is not None:
                self._schedulers = [self._schedulers] if not isinstance(self._schedulers, list) else self._schedulers
            self._optimizers = (
                [self._optimizers["optimizer"]] if not isinstance(self._optimizers, list) else self._optimizers
            )
