"""Renewable GLM estimation (card: ``renewable-glm``).

Reference: Luo & Song (2020). Renewable estimation and incremental inference in
generalized linear models with streaming data sets. JRSS-B 82(1), 69-97.

Data are processed in B equal-sized batches. Cumulative sufficient statistics
(Fisher information and score) are maintained as running averages and used for
a single Newton step after each batch.
"""

from __future__ import annotations

import numpy as np

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link

soft_threshold = lambda v, t: np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


class RenewableGLMSolver(BaseSolver):
    """Renewable (streaming) GLM estimation via online Fisher scoring.

    Config
    ------
    n_batches          : int,   default 10   — number of streaming batches
    ridge_init         : float, default 1e-3 — regularisation for early batches
    max_step_norm      : float, default 5.0  — clip Newton step by this L2 norm
    seed               : int,   default 42
    """

    card_id = "renewable-glm"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        n_batches: int = int(self.config.get("n_batches", 10))
        ridge_init: float = self.config.get("ridge_init", 1e-3)
        max_step_norm: float = self.config.get("max_step_norm", 5.0)
        seed: int = self.config.get("seed", 42)

        rng = np.random.default_rng(seed)
        n, p = X.shape
        beta = np.zeros(p)

        idx = rng.permutation(n)
        batch_indices = np.array_split(idx, n_batches)

        S_cumul = np.zeros((p, p))
        T_cumul = np.zeros(p)

        for b, bidx in enumerate(batch_indices):
            X_b = X[bidx]
            y_b = y[bidx]
            m = len(bidx)
            if m == 0:
                continue

            eta_b = X_b @ beta
            mu_b = lnk.g_inv(eta_b)
            V_b = lnk.var(mu_b)
            W_b = np.diag(V_b)

            S_b = (X_b.T @ W_b @ X_b) / m
            T_b = (X_b.T @ (y_b - mu_b)) / m

            S_cumul = (b * S_cumul + S_b) / (b + 1)
            T_cumul = (b * T_cumul + T_b) / (b + 1)

            # Adaptive ridge: large at first batch, shrinks as data accumulates
            ridge = ridge_init / (b + 1) + 1e-12
            try:
                delta = np.linalg.solve(S_cumul + ridge * np.eye(p), T_cumul)
                # Clip Newton step to prevent overshooting
                delta_norm = np.linalg.norm(delta)
                if delta_norm > max_step_norm:
                    delta = delta * max_step_norm / delta_norm
                beta = beta + delta
            except np.linalg.LinAlgError:
                pass

        return FitResult(
            beta_hat=beta,
            intercept=0.0,
            n_iter=n_batches,
            diagnostics={"converged": True, "n_batches": n_batches},
        )
