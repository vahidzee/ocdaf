# extending normflows/flow.py so that we can pass the permutation as well
import normflows as nf
import numpy as np
import torch


class NormalizingFlow(nf.NormalizingFlow):
    """
    Normalizing Flow model to approximate target distribution
    """

    def inverse(self, z, **kwargs):
        """Inverse pass through normalizing flow (latent to inputs)

        Args:
          z: Batch sampled from approximate distribution

        Returns:
          Batch of samples from target distribution
        """
        x = z
        log_dets = 0
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_det = self.flows[i].inverse(x, **kwargs)
            log_dets += log_det
        return x, log_dets

    def forward_kld(self, x, **kwargs):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, **kwargs)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def reverse_alpha_div(self, num_samples=1, alpha=1, dreg=False):
        """Alpha divergence when sampling from q

        Args:
          num_samples: Number of samples to draw
          dreg: Flag whether to use Double Reparametrized Gradient estimator, see [arXiv 1810.04152](https://arxiv.org/abs/1810.04152)

        Returns:
          Alpha divergence
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.p.log_prob(z)
        if dreg:
            w_const = torch.exp(log_p - log_q).detach()
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            nf.utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            nf.utils.set_requires_grad(self, True)
            w = torch.exp(log_p - log_q)
            w_alpha = w_const**alpha
            w_alpha = w_alpha / torch.mean(w_alpha)
            weights = (1 - alpha) * w_alpha + alpha * w_alpha**2
            loss = -alpha * torch.mean(weights * torch.log(w))
        else:
            loss = np.sign(alpha - 1) * torch.logsumexp(alpha * (log_p - log_q), 0)
        return loss

    def sample(self, num_samples=1, **kwargs):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z, **kwargs)
            log_q -= log_det
        return z, log_q
