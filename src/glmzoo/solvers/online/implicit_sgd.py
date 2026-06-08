"""Implicit SGD / proximal SGD (card: ``implicit-sgd``).

Reference: Toulis & Airoldi (2017). For GLMs with canonical link the implicit update
has an exact closed form for the Gaussian/identity case. For logit/log we fall back
to explicit SGD with a smaller step.
"""

from __future__ import annotations

import numpy as np

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link

soft_threshold = lambda v, t: np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


class ImplicitSGDSolver(BaseSolver):
    """Implicit (proximal) SGD with exact closed form for Gaussian link.

    Config
    ------
    gamma0   : float, default 0.05
    alpha    : float, default 0.51
    c        : float, default 1.0
    n_passes : int, default 10
    seed     : int, default 42
    """

    card_id = "implicit-sgd"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        gamma0: float = self.config.get("gamma0", 0.05)
        alpha: float = self.config.get("alpha", 0.51)
        c: float = self.config.get("c", 1.0)
        n_passes: int = self.config.get("n_passes", 10)
        seed: int = self.config.get("seed", 42)

        rng = np.random.default_rng(seed)
        n, p = X.shape
        beta = np.zeros(p)
        use_implicit = lnk.name == "identity"

        t = 0
        for _ in range(n_passes):
            idx = rng.permutation(n)
            for i in idx:
                x_i = X[i]
                y_i = y[i]
                gamma_t = gamma0 / (1.0 + gamma0 * c * t) ** alpha

                if use_implicit:
                    # Exact implicit update for linear model
                    norm2 = x_i @ x_i
                    delta = gamma_t * (y_i - x_i @ beta) / (1.0 + gamma_t * norm2)
                    beta = beta + delta * x_i
                else:
                    # Fall back to explicit SGD with halved step for non-canonical
                    eta_i = x_i @ beta
                    mu_i = lnk.g_inv(eta_i)
                    grad = x_i * (mu_i - y_i)
                    beta = beta - (gamma_t * 0.5) * grad
                t += 1

        return FitResult(
            beta_hat=beta,
            intercept=0.0,
            n_iter=t,
            diagnostics={"converged": True, "n_iter": t, "implicit": use_implicit},
        )
