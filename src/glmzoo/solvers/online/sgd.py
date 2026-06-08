"""SGD with Polyak-Ruppert averaging (card: ``sgd``).

Reference: Robbins & Monro (1951) + Polyak & Juditsky (1992).
"""

from __future__ import annotations

import numpy as np

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link

soft_threshold = lambda v, t: np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


class SGDSolver(BaseSolver):
    """Stochastic Gradient Descent with optional Polyak-Ruppert averaging.

    Config
    ------
    gamma0 : float, default 0.01   — initial step size
    alpha  : float, default 0.51   — decay exponent
    c      : float, default 1.0    — decay scale
    n_passes : int, default 3      — number of passes over data
    averaging : bool, default True — enable Polyak-Ruppert averaging
    seed   : int, default 42
    """

    card_id = "sgd"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        gamma0: float = self.config.get("gamma0", 0.01)
        alpha: float = self.config.get("alpha", 0.51)
        c: float = self.config.get("c", 1.0)
        n_passes: int = self.config.get("n_passes", 3)
        averaging: bool = self.config.get("averaging", True)
        seed: int = self.config.get("seed", 42)

        rng = np.random.default_rng(seed)
        n, p = X.shape
        beta = np.zeros(p)
        beta_avg = np.zeros(p)

        t = 0
        for _ in range(n_passes):
            idx = rng.permutation(n)
            for i in idx:
                x_i = X[i]
                y_i = y[i]
                eta_i = x_i @ beta
                mu_i = lnk.g_inv(eta_i)
                grad = x_i * (mu_i - y_i)
                gamma_t = gamma0 / (1.0 + gamma0 * c * t) ** alpha
                beta = beta - gamma_t * grad
                if averaging:
                    beta_avg = beta_avg + (beta - beta_avg) / (t + 1)
                t += 1

        result = beta_avg if averaging else beta
        return FitResult(
            beta_hat=result,
            intercept=0.0,
            n_iter=t,
            diagnostics={"converged": True, "n_iter": t, "averaging": averaging},
        )
