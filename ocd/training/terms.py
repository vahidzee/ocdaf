from lightning_toolbox import CriterionTerm
from ocd.models.utils import log_prob
import torch
import lightning
import typing as th
import dycode as dy


class OrderedLikelihoodTerm(CriterionTerm):
    def __init__(
        self,
        name: th.Optional[str] = "nll",
        factor: th.Optional[th.Union[float, dy.FunctionDescriptor]] = None,
        scale_factor: th.Optional[str] = None,
        term_function: th.Optional[dy.FunctionDescriptor] = None,
        factor_application: str = "multiply",  # multiply or add
        **kwargs,  # function description dictionary
    ):
        super().__init__(
            name=name,
            factor=factor,
            scale_factor=scale_factor,
            term_function=term_function,
            factor_application=factor_application,
            **kwargs,
        )

    def __call__(
        self,
        batch: th.Any = None,  # one-hot encoded batch
        original_batch: th.Any = None,  # not one-hot encoded
        # training module (.model to access model)
        training_module: lightning.LightningModule = None,
        **kwargs,
    ):
        # get model output
        interventional: bool = isinstance(
            batch, (list, tuple)
        )  # whether to use interventional or non-interventional loss
        interventional_batch_size: int = batch[1].shape[0] if interventional else 0

        if interventional:
            batch = torch.cat(batch, dim=0)
            original_batch = torch.cat(original_batch, dim=0)

        training_module.term_batch = batch
        training_module.term_original_batch = original_batch

        model_output = training_module(batch)

        # get likelihoods
        log_likelihoods = log_prob(
            probas=model_output,
            cov_features=training_module.model.in_covariate_features,
            categories=original_batch,
            reduce=False,
        )

        # add interventional support
        if interventional:
            int_log_likelihoods = log_likelihoods[-interventional_batch_size:]
            log_likelihoods = log_likelihoods[:-interventional_batch_size]

            # get the argmin over each sample in the interventional batch
            argmin = torch.argmin(int_log_likelihoods, dim=1)
            # mask the log likelihoods with the argmin with 0
            int_log_likelihoods = torch.where(
                argmin[:, None] == torch.arange(
                    int_log_likelihoods.shape[1], device=argmin.device)[None, :],
                torch.zeros_like(int_log_likelihoods),
                int_log_likelihoods,
            )
            # add the interventional log likelihoods to the non-interventional log likelihoods
            log_likelihoods = torch.cat(
                [log_likelihoods, int_log_likelihoods], dim=0)

        return -log_likelihoods.sum(dim=1).mean()
