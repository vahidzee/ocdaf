import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from ..logging import LoggingCallback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import typing as th

from collections import defaultdict


class LoggingCallback(LoggingCallback):
    """
    This is a callback used for logging the input and outputs of the training process
    the logs from this callback are used to generate plots for the other visualizing callbacks.

    All the values that are latched in the criterion are logged in the all_logged_values dict
    """

    def __init__(self) -> None:
        super().__init__()
