import torch
import typing as th
from .ordered_linear import OrderedLinear


class AutoRegressiveDensityEstimator1D(OrderedLinear):
    """
    Based on an input doubly stochastic matrix P
    it calculates the corresponding autoregressive mask
    and returns the corresponding logits.

    Then using softmax over all the categories on the logits, it calculates the probability of each category.
    Hence, out_cov_features[i] yields the number of categories for the ith covariate.
    """

    def __init__(
        self,
        in_cov_features: th.List[int],
        out_cov_features: th.List[int],
        auto_connection: bool = True,
        bias: bool = True,
        dtype: th.Optional[torch.dtype] = None,
        device: th.Optional[torch.device] = None,
        mask_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            in_cov_features=in_cov_features,
            out_cov_features=out_cov_features,
            bias=bias,
            device=device,
            dtype=dtype,
            auto_connection=auto_connection,
            mask_dtype=mask_dtype,
        )

    def forward(self, inputs, P) -> torch.Tensor:
        """
        Calculates the logits for [n] categorical covariates.
        The end result is (batch_size, sum of [out_cov_features])
        because for each single input all of the logits for the category are concatenated.

        Args:
            inputs: (batch_size, sum of [in_cov_features])
            P: a permutation matrix to order the autoregressive mask
        Returns:
            probas: (batch_size, sum of [out_cov_features])
        """
        logits, P = super().forward(inputs, P)
        logits = torch.exp(logits)
        # Create a blocked diagonal matrix according to a list of block sizes
        probas = logits / (
            logits
            @ torch.block_diag(*[torch.ones((x, x), device=inputs.device) for x in self.out_cov_features.tolist()])
        )
        return probas
