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

    def forward(self, inputs, P, eps=1e-8) -> torch.Tensor:
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
        # add logits with an epsilon to avoid log(0)
        logits = logits + eps
        # take logsoftmax over the covariate cagetories
        # first take the cumulative sum of the out_cov_features (add a zero at the beginning)
        cumsum_out_cov_features = torch.cat(
            (torch.zeros(1, device=inputs.device),
             torch.cumsum(self.out_cov_features, dim=0))
        ).int()

        log_softmax = [
            torch.log_softmax(
                logits[:, cumsum_out_cov_features[i]: cumsum_out_cov_features[i + 1]], dim=1)
            for i in range(len(self.out_cov_features))
        ]
        log_softmax = torch.cat(log_softmax, dim=1)
        return log_softmax
