"""FOBOS / proximal SGD (card: ``fobos``).

Reference: Duchi & Singer (2009). Efficient online and batch learning using forward
backward splitting. JMLR 10, 2899-2934.
"""

from __future__ import annotations

import numpy as np

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link

soft_threshold = lambda v, t: np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


class FOBOSSolver(BaseSolver):
    """Forward-Backward Splitting (FOBOS) with L1 proximal step.

    Config
    ------
    lam      : float, default 0.1  — L1 regularisation weight
    gamma0   : float, default 0.1  — initial step size
    n_passes : int, default 10
    seed     : int, default 42
    """

    card_id = "fobos"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam: float = self.config.get("lam", 0.1)
        gamma0: float = self.config.get("gamma0", 0.1)
        n_passes: int = self.config.get("n_passes", 10)
        seed: int = self.config.get("seed", 42)

        rng = np.random.default_rng(seed)
        n, p = X.shape
        beta = np.zeros(p)

        t = 0
        for _ in range(n_passes):
            idx = rng.permutation(n)
            for i in idx:
                x_i = X[i]
                y_i = y[i]
                eta_i = x_i @ beta
                mu_i = lnk.g_inv(eta_i)
                grad = x_i * (mu_i - y_i)
                gamma_t = gamma0 / np.sqrt(t + 1)
                beta_half = beta - gamma_t * grad
                beta = soft_threshold(beta_half, gamma_t * lam)
                t += 1

        return FitResult(
            beta_hat=beta,
            intercept=0.0,
            n_iter=t,
            diagnostics={"converged": True, "n_iter": t},
        )
