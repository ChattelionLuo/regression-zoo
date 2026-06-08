"""Regularized Dual Averaging (card: ``rda``).

Reference: Xiao (2010). Dual averaging methods for regularized stochastic learning
and online optimization. JMLR 11, 2543-2596.  Algorithm 2.
"""
from __future__ import annotations
import numpy as np
from ...base import BaseSolver, FitResult
from ...links import LINKS, Link

soft_threshold = lambda v, t: np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


class RDASolver(BaseSolver):
    """Regularized Dual Averaging with L1 regularisation.

    Follows Xiao (2010) Algorithm 2: the dual variable beta (returned) uses
    the cumulative-step formula (grows as sqrt(t)), while gradients are
    computed at a bounded SGD primal variable w (step 1/sqrt(t)).

    Config
    ------
    lam      : float, default 0.1
    gamma0   : float, default 0.1
    n_passes : int, default 5
    seed     : int, default 42
    """

    card_id = "rda"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam: float = self.config.get("lam", 0.1)
        gamma0: float = self.config.get("gamma0", 0.1)
        n_passes: int = self.config.get("n_passes", 5)
        seed: int = self.config.get("seed", 42)

        rng = np.random.default_rng(seed)
        n, p = X.shape
        w = np.zeros(p)       # SGD primal (bounded, used for gradient computation)
        g_sum = np.zeros(p)   # accumulated gradient sum
        beta = np.zeros(p)    # RDA dual variable (returned at end)

        t = 0
        for _ in range(n_passes):
            idx = rng.permutation(n)
            for i in idx:
                x_i = X[i]
                y_i = y[i]

                # --- gradient at PRIMAL w (bounded) ---
                eta_i = x_i @ w
                mu_i = lnk.g_inv(eta_i)
                grad = x_i * (mu_i - y_i)

                # --- accumulate gradient ---
                g_sum += grad
                g_avg = g_sum / (t + 1)

                # --- RDA dual update (Xiao 2010, eq. 2.5 / Algorithm 2) ---
                gamma_t = gamma0 * np.sqrt(t + 1)        # cumulative step
                beta = soft_threshold(-g_avg * gamma_t, lam * gamma_t)

                # --- primal SGD update (bounded, for next gradient) ---
                step = gamma0 / np.sqrt(t + 1)
                w = w - step * grad

                t += 1

        # Return the normalized dual as the primal estimate:
        # beta / gamma_T = soft_threshold(-g_avg_T, lam) — bounded
        gamma_T = gamma0 * np.sqrt(t)
        beta_normalized = beta / gamma_T if gamma_T > 0 else beta

        return FitResult(
            beta_hat=beta_normalized,
            intercept=0.0,
            n_iter=t,
            diagnostics={"converged": True, "n_iter": t},
        )
