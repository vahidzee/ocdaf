
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, EPOCH_DEPRECATION_WARNING
import warnings

class ReduceLROnIncrease(ReduceLROnPlateau):

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch
        
        if self.is_worse(current, self.best):
            self.num_bad_epochs += 1
        elif self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
    
    def is_worse(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. + self.threshold
            return a > best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a > best + self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a < best - self.threshold