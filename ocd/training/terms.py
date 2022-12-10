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
        batch_sizes = (
            [batch[i].shape[0] for i in range(len(batch))] if isinstance(batch, (list, tuple)) else [batch.shape[0]]
        )

        if len(batch_sizes) > 1:
            batch = torch.cat(batch, dim=0)
            original_batch = torch.cat(original_batch, dim=0)

        # get logits
        model_output = training_module(batch)
        # get likelihoods
        log_likelihoods = log_prob(
            probas=model_output,
            cov_features=training_module.model.in_covariate_features,
            categories=original_batch,
            reduce=False,
        )

        # split likelihoods into batches
        log_likelihoods = torch.split(log_likelihoods, batch_sizes, dim=0)

        # for each interventional batch, put aside the logit with the lowest likelihood on average
        # and set it to 0
        results = -log_likelihoods[0].sum(dim=1).mean()  # observational data
        intervention_idx = (
            torch.stack(log_likelihoods[1:]).mean(dim=1).argmin(dim=1)
        )  # probable intervention index in each interventional batch

        # create a mask to set the logit with the lowest likelihood to 0
        mask = intervention_idx.reshape(-1, 1) == torch.arange(
            log_likelihoods[0].shape[-1], device=results.device
        ).reshape(1, -1)
        mask = mask.repeat_interleave(
            torch.tensor(batch_sizes[1:]).to(device=results.device), dim=0
        )  # repeat each row of mask by the number of samples in the corresponding batch

        # mask the logit with the lowest likelihood
        interventional_log_likelihoods = torch.cat(
            log_likelihoods[1:]
        )  # concatenate interventional batches into one tensor to process them simoultaneously
        interventional_log_likelihoods = torch.where(
            mask,
            torch.zeros_like(interventional_log_likelihoods, device=results.device),
            interventional_log_likelihoods,
        )
        results = results - interventional_log_likelihoods.sum(dim=1).mean()
        return results
