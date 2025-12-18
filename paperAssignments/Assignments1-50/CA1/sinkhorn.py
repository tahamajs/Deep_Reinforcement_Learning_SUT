import math
from typing import Tuple

import torch
import torch.nn as nn


class AnnealedSinkhornLoss(nn.Module):
    """
    Annealed Sinkhorn Divergence loss with implicit differentiation approximation.

    Args:
        n_iters: number of Sinkhorn iterations (for fixed-point computation).
        eps_start: starting epsilon for annealing (large -> smooth)
        eps_end: final epsilon (small -> close to Wasserstein)
        decay_steps: number of steps over which to anneal epsilon
        clamp_eps_min: minimum epsilon allowed for numerical stability
    """

    def __init__(
        self,
        n_iters: int = 20,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        decay_steps: int = 100000,
        clamp_eps_min: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_iters = int(n_iters)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)
        self.decay_steps = int(decay_steps)
        self.clamp_eps_min = float(clamp_eps_min)
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def get_epsilon(self) -> float:
        """Return current epsilon following exponential decay schedule."""
        step = float(self._step.item())
        progress = min(1.0, step / max(1.0, float(self.decay_steps)))
        if self.eps_start <= 0 or self.eps_end <= 0:
            return max(self.eps_end, self.clamp_eps_min)
        alpha = self.eps_end / self.eps_start
        eps = self.eps_start * (alpha**progress)
        return float(max(eps, self.clamp_eps_min))

    def step_annealing(self, n: int = 1) -> None:
        """Advance internal step counter (call once per optimizer step)."""
        self._step += n

    @staticmethod
    def compute_cost_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute squared Euclidean cost matrix between particles.

        Args:
            x: (B, N, D)
            y: (B, M, D)
        Returns:
            C: (B, N, M)
        """
        # broadcasting difference
        # (B, N, 1, D) - (B, 1, M, D) -> (B, N, M, D)
        diff = x.unsqueeze(2) - y.unsqueeze(1)
        return torch.sum(diff * diff, dim=-1)

    def sinkhorn_log_potentials(
        self, C: torch.Tensor, eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log-potentials f and g via fixed-point iterations in log-domain.

        C: (B, N, M)
        returns f: (B, N), g: (B, M)
        """
        B, N, M = C.shape
        device = C.device

        # initialize log-potentials (f = eps * log u, g = eps * log v) to zeros
        f = torch.zeros(B, N, device=device, dtype=C.dtype)
        g = torch.zeros(B, M, device=device, dtype=C.dtype)

        # uniform log-weights: log(1/N) and log(1/M)
        log_mu = -math.log(N)
        log_nu = -math.log(M)

        # iterate
        for _ in range(self.n_iters):
            # M = (g_j - C_ij) / eps  -> shape (B, N, M)
            Mmat = (g.unsqueeze(1) - C) / eps
            # f_i = eps * (log_mu - logsumexp(M, dim=2))
            f = eps * (log_mu - torch.logsumexp(Mmat, dim=2))

            # M = (f_i - C_ij) / eps -> shape (B, N, M)
            Mmat = (f.unsqueeze(2) - C) / eps
            g = eps * (log_nu - torch.logsumexp(Mmat, dim=1))

        return f, g

    def get_transport_cost(self, C: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute regularized transport cost <P, C> using implicit potentials.

        We run the Sinkhorn iterations with no grad to obtain f,g (implicit), then
        reconstruct P = exp((f + g - C) / eps) and compute sum(P * C).

        Returns mean over batch scalar.
        """
        # implicit fixed point (no grad through iterations)
        with torch.no_grad():
            f, g = self.sinkhorn_log_potentials(C, eps)

        # reconstruct log P
        log_P = (f.unsqueeze(2) + g.unsqueeze(1) - C) / eps
        P = torch.exp(log_P)

        # ensure numerical safety
        # normalize P to have marginals close to uniform (optional)
        cost = torch.sum(P * C, dim=(1, 2))
        return cost.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Sinkhorn divergence S_eps(pred, target).

        pred: (B, N, D)
        target: (B, M, D)
        Returns scalar loss.
        """
        eps = self.get_epsilon()

        C_xy = self.compute_cost_matrix(pred, target)
        C_xx = self.compute_cost_matrix(pred, pred)
        C_yy = self.compute_cost_matrix(target, target)

        term_xy = self.get_transport_cost(C_xy, eps)
        term_xx = self.get_transport_cost(C_xx, eps)
        term_yy = self.get_transport_cost(C_yy, eps)

        loss = term_xy - 0.5 * (term_xx + term_yy)
        return loss


if __name__ == "__main__":
    # quick smoke test
    loss_fn = AnnealedSinkhornLoss(n_iters=10)
    a = torch.randn(2, 8, 1)
    b = a.clone()
    print("S(a,a) =", loss_fn(a, b).item())







