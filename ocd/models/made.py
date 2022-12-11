import torch
import typing as th
from .layers import OrderedBlock, AutoRegressiveDensityEstimator1D


class SingleMaskedBlockMADE(torch.nn.Module):
    """
    A Masked Autoregressive Density Estimator (MADE) that can work with multiple domain data
    this model takes in a list of categorical variables (one-hot encoded)
    and given a doubly stochastic matrix P (permutation matrix), it will create masks according to
    that permutation matrix and then apply the masks to each of the layers to obtain the final output in an autoregressive fashion

    The model is defined as follows:

    Args:
        in_covariate_features: a list of integers that specifies the number of features for each covariate
        hidden_features_per_covariate: a list of lists of integers that specifies the number of hidden features for each layer for each covariate
        bias: whether to use bias in the layers
        activation: the activation function to use in the layers
        activation_args: the arguments to pass to the activation function
        batch_norm: whether to use batch normalization in the layers
        batch_norm_args: the arguments to pass to the batch normalization layers
        seed: the seed to use for the random number generator
        device: the device to use for the model
        dtype: the dtype to use for the model

        Example of made construction:
        model = SingleMaskedBlockMADE(
            in_covariate_features=[3, 4, 2, 3],
            hidden_features_per_covariate=[[10, 20, 30, 10],
                                           [20, 30, 10, 5],
                                           [40, 30, 10, 7]],
            bias=True,
            activation=torch.nn.ReLU,
        )

        This MADE takes in 4 covariates with 3, 4, 2, and 3 features respectively
        and in the output after calling forward it will output (3 + 4 + 2 + 3) probability values
        - The first three will indicate the probability of the first covariate being 0, 1, or 2
        - The next four will indicate the probability of the second covariate being 0, 1, 2, or 3
        - The next two will indicate the probability of the third covariate being 0 or 1
        - The last three will indicate the probability of the fourth covariate being 0, 1, or 2

    """

    def __init__(
        self,
        in_covariate_features: th.List[int],
        hidden_features_per_covariate: th.List[th.List[int]],
        # layers
        bias: bool = True,
        # Properties of the activations used in each layer
        activation: th.Optional[str] = "torch.nn.ReLU",
        activation_args: th.Optional[dict] = None,
        # Properties of the batch normalization layers
        batch_norm: bool = True,
        batch_norm_args: th.Optional[dict] = None,
        # general
        seed: int = 0,
        device: th.Optional[torch.device] = None,
        dtype: th.Optional[torch.dtype] = None,
        # grad safety
        safe_grad_hook: str = "lambda grad: torch.where(torch.isnan(grad) + torch.isinf(grad), torch.zeros_like(grad), grad)",
    ) -> None:
        super().__init__()
        self.L = len(hidden_features_per_covariate)
        self.layers = torch.nn.Sequential(
            *[
                OrderedBlock(
                    in_cov_features=in_covariate_features if not i else hidden_features_per_covariate[
                        i - 1],
                    out_cov_features=hidden_features_per_covariate[i],
                    bias=bias,
                    activation=activation,
                    activation_args=activation_args,
                    batch_norm=batch_norm,
                    batch_norm_args=batch_norm_args,
                    device=device,
                    dtype=dtype,
                )
                for i in range(self.L)
            ]
        )

        self.density_estimator = AutoRegressiveDensityEstimator1D(
            in_cov_features=hidden_features_per_covariate[-1],
            out_cov_features=in_covariate_features,
            bias=bias,
            device=device,
            dtype=dtype,
            auto_connection=False,
        )

        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        self.safe_grad_hook = safe_grad_hook

    def forward(
        self,
        inputs,
        perm: th.Union[th.List[int], torch.Tensor],
    ):
        # Takes in inputs of shape (batch_size, sum of [input features per covariate])
        # and either a permutation of the covariates or a permanent matrix for permuting the covariates
        # If the permutation is a list, the gradients are off, otherwise, they will flow through the permutation matrix

        results, perm = self.layers((inputs, perm))
        results = self.density_estimator(results, perm)

        # if results.requires_grad and safe_grad:
        #     results.register_hook(self.safe_grad_hook_function)
        return results
