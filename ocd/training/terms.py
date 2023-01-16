from lightning_toolbox import CriterionTerm
import torch
import lightning
import typing as th
import dypy as dy


class TrainingTerm(CriterionTerm):
    def __init__(
        self,
        name: th.Optional[str] = "losses",
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
        batch: th.Any = None,  # a batch of data
        # training module (.model to access model)
        training_module: lightning.LightningModule = None,
        **kwargs,
    ):
        try:
            if training_module.get_phase() == "expectation":
                # Maximize the whole ELBO
                if training_module.model.elementwise_perm:
                    res = training_module.model(
                        batch,
                        soft=True,
                        training_module=training_module,
                        return_noise_prob=True,
                        return_latent_permutation=training_module.log_input_outputs,
                    )
                    all_log_probs = res["log_prob"]
                    # ([batch_size], [batch_size], [batch_size]])
                    loss = all_log_probs  # + log_noise_prob
                    return -loss.mean(dim=0)
                else:
                    res = training_module.model(
                        batch,
                        soft=True,
                        training_module=training_module,
                        return_latent_permutation=training_module.log_input_outputs,
                    )
                    # ([batch_size, num_permutations], [num_permutations], [num_permutations])
                    all_log_probs = res["log_prob"]
                    loss = all_log_probs  # + log_noise_prob
                    return -loss.mean(dim=0)

            elif training_module.get_phase() == "maximization":
                if training_module.model.elementwise_perm:
                    res = training_module.model(
                        batch,
                        training_module=training_module,
                        soft=training_module.use_soft_on_maximization,
                        return_noise_prob=False,
                        return_prior=False,
                        return_latent_permutation=training_module.log_input_outputs,
                    )
                    all_log_probs = res["log_prob"]
                    # ([batch_size, ])
                    return -all_log_probs.mean(dim=0)
                else:
                    res = training_module.model(
                        batch,
                        training_module=training_module,
                        soft=training_module.use_soft_on_maximization,
                        return_noise_prob=False,
                        return_prior=False,
                        return_latent_permutation=training_module.log_input_outputs,
                    )
                    all_log_probs = res["log_prob"]
                    # ([batch_size, num_permutations])
                    all_log_probs = all_log_probs.mean(dim=0)  # [num_permutations]
                return -all_log_probs.mean(dim=0)
            else:
                raise NameError(f"Phase {training_module.get_phase()} non-existent.")
        finally:
            if training_module.log_input_outputs:
                training_module.log_new_input_outputs(res)
