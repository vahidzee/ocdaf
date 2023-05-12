import torch
from ocd.models.affine_flow import AffineFlow
import typing as th
from ocd.models.permutation import LearnablePermutation, gumbel_log_prob, LegacyLearnablePermutation
import dypy as dy
from lightning_toolbox import TrainingModule
from ocd.models.permutation.module import PERMUTATION_TYPE_OPTIONS
import warnings


class OCDAF(torch.nn.Module):
    def __init__(
        self,
        # architecture
        in_features: th.Optional[th.Union[th.List[int], int]] = None,
        layers: th.List[th.Union[th.List[int], int]] = None,
        populate_features: bool = False,  # if True, values in layers are multiplied by in_features
        residual: bool = False,
        bias: bool = True,
        activation: th.Optional[str] = "torch.nn.LeakyReLU",
        activation_args: th.Optional[dict] = None,
        batch_norm: bool = False,
        batch_norm_args: th.Optional[dict] = None,
        # additional flow args
        additive: bool = False,
        scale_transform: bool = False,
        scale_transform_s_args: th.Optional[dict] = None,
        scale_transform_t_args: th.Optional[dict] = None,
        share_parameters: bool = False,  # share parameters between scale and shift
        num_transforms: int = 1,
        # base distribution
        base_distribution: th.Union[torch.distributions.Distribution, str] = "torch.distributions.Normal",
        base_distribution_args: dict = dict(loc=0.0, scale=1.0),  # type: ignore
        # ordering
        ordering: th.Optional[torch.IntTensor] = None,
        reversed_ordering: bool = False,
        use_permutation: bool = True,
        permutation_learner_cls: th.Optional[str] = "ocd.models.permutation.LearnablePermutation",
        permutation_learner_args: th.Optional[dict] = None,
        # general args
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if in_features is None:
            warnings.warn("in_features is None, this might cause issues")
            in_features = 3

        # populate features if necessary
        in_features = in_features if isinstance(in_features, int) else len(in_features)
        if populate_features:
            for i in range(len(layers)):
                if isinstance(layers[i], int):
                    layers[i] *= in_features
                else:
                    for j in range(len(layers[i])):
                        layers[i][j] *= in_features

        # get an IntTensor of 1 to N
        self.flow = AffineFlow(
            base_distribution=base_distribution,
            base_distribution_args=base_distribution_args,
            in_features=in_features,
            layers=layers,
            residual=residual,
            bias=bias,
            activation=activation,
            activation_args=activation_args,
            batch_norm=batch_norm,
            batch_norm_args=batch_norm_args,
            additive=additive,
            scale_transform=scale_transform,
            scale_transform_s_args=scale_transform_s_args,
            scale_transform_t_args=scale_transform_t_args,
            share_parameters=share_parameters,
            num_transforms=num_transforms,
            ordering=ordering,
            reversed_ordering=reversed_ordering,
            device=device,
            dtype=dtype,
        )

        if use_permutation:
            self.permutation_model = dy.get_value(permutation_learner_cls)(
                num_features=in_features if isinstance(in_features, int) else len(in_features),
                device=device,
                dtype=dtype,
                **(permutation_learner_args or dict()),
            )
        else:
            self.permutation_model = None

    def forward(
        self,
        inputs: torch.Tensor,
        # permutation
        permutation_type: th.Optional[PERMUTATION_TYPE_OPTIONS] = None,
        permute: bool = True,
        # return
        return_log_prob: bool = True,
        return_noise_prob: bool = False,
        return_prior: bool = False,
        return_latent_permutation: bool = False,
        # args for dynamic methods
        training_module: th.Optional["TrainingModule"] = None,
        **kwargs
    ):
        results = {}

        # sample latent permutation
        latent_permutation, gumbel_noise = None, None
        if self.permutation_model is not None and permute:
            permutation_results, gumbel_noise = self.permutation_model(
                batch_size=inputs.shape[0],
                device=inputs.device,
                permutation_type=permutation_type,
                return_noise=True,
                training_module=training_module,
                **kwargs,
            )

        if (
            self.permutation_model is not None
            and self.permutation_model.permutation_type != "hybrid-sparse-map-simulator"
        ):
            # This is an element-wise input whereby for each input
            # we have one permutation
            latent_permutation = permutation_results["perm_mat"]
            log_prob = self.flow.log_prob(inputs, perm_mat=latent_permutation)

            if training_module is not None:
                # Save for callbacks
                training_module.remember(
                    elementwise=True, elementwise_input=inputs, elementwise_perm_mat=latent_permutation
                )

                if "soft_perm_mats" in permutation_results:
                    training_module.remember(permutation_to_display=permutation_results["soft_perm_mats"])
                else:
                    training_module.remember(permutation_to_display=latent_permutation)

                training_module.remember(log_prob_to_display=log_prob)

        elif (
            self.permutation_model is not None
            and self.permutation_model.permutation_type == "hybrid-sparse-map-simulator"
        ):
            # This is the case where it is not Elementwise, for example,
            # this might happen for the hybrid-sparse-map-simulation
            soft_perm_mat = permutation_results["soft_perm_mat"]
            hard_perm_mat = permutation_results["hard_perm_mat"]
            score_grid = permutation_results["score_grid"]
            if score_grid.shape[0] != inputs.shape[0]:
                assert (
                    inputs.shape[0] % score_grid.shape[0] == 0
                ), "batch_size must be divisible by score_grid.shape[0]"
                score_grid = score_grid.repeat(inputs.shape[0] // score_grid.shape[0], 1)
            inputs_repeated = torch.repeat_interleave(inputs, repeats=hard_perm_mat.shape[0], dim=0)
            hard_perm_mat_repeated = hard_perm_mat.repeat(inputs.shape[0], 1, 1)
            all_log_probs = self.flow.log_prob(inputs_repeated, perm_mat=hard_perm_mat)
            log_prob_grid = all_log_probs.reshape(inputs.shape[0], hard_perm_mat.shape[0])
            log_prob = torch.sum(log_prob_grid * score_grid, dim=1)

            if training_module is not None:
                training_module.remember(
                    elementwise=False,
                    log_prob_to_display=log_prob,
                    permutation_to_display=soft_perm_mat,
                    elementwise_input=inputs,
                    elementwise_perm_mat=hard_perm_mat_repeated,
                )

        elif self.permutation_model is None or not permute:
            log_prob = self.flow.log_prob(inputs, perm_mat=latent_permutation)

            if training_module is not None:
                training_module.remember(elementwise_input=inputs)

        # return log_prob, noise_prob, prior (if requested)
        ret = dict(log_prob=log_prob) if return_log_prob else dict()
        if return_noise_prob:
            ret["noise_prob"] = gumbel_log_prob(gumbel_noise) if gumbel_noise is not None else 0
        if return_prior:
            raise NotImplementedError("Haven't implemented prior yet.")
        if return_latent_permutation:
            ret["latent_permutation"] = permutation_results

        # TODO: remove this!
        if "scores" in permutation_results:
            ret["scores"] = permutation_results["scores"]
        return ret
