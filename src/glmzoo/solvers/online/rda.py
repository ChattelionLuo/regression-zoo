"""Regularized Dual Averaging (card: ``rda``).

Reference: Xiao (2010). Dual averaging methods for regularized stochastic learning
and online optimization. JMLR 11, 2543-2596.  Eq. (2.5).
"""

from __future__ import annotations

import numpy as np

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link

soft_threshold = lambda v, t: np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


class RDASolver(BaseSolver):
    """Regularized Dual Averaging with L1 regularisation.

    The closed-form update (Xiao 2010, eq. 2.5) is:

        beta_{t+1} = soft_threshold(-g_avg * gamma_t, lam * gamma_t)

    where g_avg is the running mean of observed gradients and
    gamma_t = gamma0 * sqrt(t+1) is the cumulative step scale.

    Config
    ------
    lam      : float, default 0.1
    gamma0   : float, default 0.1
    n_passes : int, default 3
    seed     : int, default 42
    """

    card_id = "rda"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam: float = self.config.get("lam", 0.1)
        gamma0: float = self.config.get("gamma0", 0.1)
        n_passes: int = self.config.get("n_passes", 3)
        seed: int = self.config.get("seed", 42)

        rng = np.random.default_rng(seed)
        n, p = X.shape
        beta = np.zeros(p)
        g_avg = np.zeros(p)  # running average of gradients

        t = 0
        for _ in range(n_passes):
            idx = rng.permutation(n)
            for i in idx:
                x_i = X[i]
                y_i = y[i]
                eta_i = x_i @ beta
                mu_i = lnk.g_inv(eta_i)
                grad = x_i * (mu_i - y_i)
                g_avg = (t * g_avg + grad) / (t + 1)
                # cumulative step scale: gamma_t = gamma0 * sqrt(t+1)
                gamma_t = gamma0 * np.sqrt(t + 1)
                beta = soft_threshold(-g_avg * gamma_t, lam * gamma_t)
                t += 1

        return FitResult(
            beta_hat=beta,
            intercept=0.0,
            n_iter=t,
            diagnostics={"converged": True, "n_iter": t},
        )
