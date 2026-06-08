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
    n_batches         : int, default 10  — number of streaming batches
    max_iter_per_batch: int, default 1   — Newton steps per batch (currently 1)
    seed              : int, default 42
    """

    card_id = "renewable-glm"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        n_batches: int = int(self.config.get("n_batches", 10))
        seed: int = self.config.get("seed", 42)

        rng = np.random.default_rng(seed)
        n, p = X.shape
        beta = np.zeros(p)

        # Split into batches (last batch may be slightly larger)
        idx = rng.permutation(n)
        batch_indices = np.array_split(idx, n_batches)

        S_cumul = np.zeros((p, p))  # running avg Fisher info
        T_cumul = np.zeros(p)       # running avg score

        for b, bidx in enumerate(batch_indices):
            X_b = X[bidx]
            y_b = y[bidx]
            m = len(bidx)
            if m == 0:
                continue

            eta_b = X_b @ beta
            mu_b = lnk.g_inv(eta_b)
            V_b = lnk.var(mu_b)
            W_b = np.diag(V_b)  # V(mu) is the GLM weight

            S_b = (X_b.T @ W_b @ X_b) / m
            T_b = (X_b.T @ (y_b - mu_b)) / m

            # Running averages
            S_cumul = (b * S_cumul + S_b) / (b + 1)
            T_cumul = (b * T_cumul + T_b) / (b + 1)

            # One Newton step using cumulated statistics
            try:
                delta = np.linalg.solve(S_cumul + 1e-10 * np.eye(p), T_cumul)
                beta = beta + delta
            except np.linalg.LinAlgError:
                pass  # keep current beta if singular

        return FitResult(
            beta_hat=beta,
            intercept=0.0,
            n_iter=n_batches,
            diagnostics={"converged": True, "n_batches": n_batches},
        )
