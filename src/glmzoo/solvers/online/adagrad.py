"""AdaGrad solver (card: ``adagrad``).

Reference: Duchi, Hazan & Singer (2011). Adaptive subgradient methods for online
learning and stochastic optimization. JMLR 12, 2121-2159.
"""

from __future__ import annotations

import numpy as np

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link

soft_threshold = lambda v, t: np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


class AdaGradSolver(BaseSolver):
    """Adaptive Gradient algorithm with per-coordinate learning rates.

    Config
    ------
    eta      : float, default 0.1   — global step size
    eps      : float, default 1e-8  — numerical stability constant
    n_passes : int, default 10
    seed     : int, default 42
    """

    card_id = "adagrad"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        eta: float = self.config.get("eta", 0.1)
        eps: float = self.config.get("eps", 1e-8)
        n_passes: int = self.config.get("n_passes", 10)
        seed: int = self.config.get("seed", 42)

        rng = np.random.default_rng(seed)
        n, p = X.shape
        beta = np.zeros(p)
        G = np.zeros(p)  # accumulated squared gradients

        t = 0
        for _ in range(n_passes):
            idx = rng.permutation(n)
            for i in idx:
                x_i = X[i]
                y_i = y[i]
                eta_i = x_i @ beta
                mu_i = lnk.g_inv(eta_i)
                grad = x_i * (mu_i - y_i)
                G = G + grad ** 2
                beta = beta - (eta / np.sqrt(G + eps)) * grad
                t += 1

        return FitResult(
            beta_hat=beta,
            intercept=0.0,
            n_iter=t,
            diagnostics={"converged": True, "n_iter": t},
        )
