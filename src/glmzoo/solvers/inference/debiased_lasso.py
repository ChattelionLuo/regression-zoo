"""Debiased / Desparsified Lasso (card: ``debiased-lasso``).

References:
  - van de Geer, Bühlmann, Ritov & Dezeure (2014). On asymptotically optimal
    confidence regions and tests for high-dimensional models. Ann. Statist. 42.
  - Zhang & Zhang (2014). Confidence intervals for low dimensional parameters in
    high dimensional linear regression. JRSS-B 76(1), 217-242.
"""

from __future__ import annotations

import numpy as np

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link

soft_threshold = lambda v, t: np.sign(v) * np.maximum(np.abs(v) - t, 0.0)

try:
    from sklearn.linear_model import Lasso as _Lasso
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class DebiasedLassoSolver(BaseSolver):
    """Debiased Lasso with nodewise regression for standard errors.

    Config
    ------
    lam      : float, default 0.1  — Lasso penalty for primary fit
    lam_node : float, default 0.1  — Lasso penalty for nodewise regressions
    """

    card_id = "debiased-lasso"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        if not _HAS_SKLEARN:
            raise ImportError("sklearn is required for DebiasedLassoSolver")

        lnk = self._resolve_link(link)
        lam: float = self.config.get("lam", 0.1)
        lam_node: float = self.config.get("lam_node", 0.1)

        n, p = X.shape

        # ── Step 1: Fit base lasso ────────────────────────────────────────────
        lasso = _Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
        lasso.fit(X, y)
        beta_hat = lasso.coef_.copy()

        # ── Step 2: Nodewise lasso to build precision-matrix surrogate Theta ──
        cols = np.arange(p)
        Theta = np.zeros((p, p))
        tau2 = np.zeros(p)

        for j in range(p):
            others = cols[cols != j]
            X_j = X[:, j]
            X_nj = X[:, others]
            node_lasso = _Lasso(alpha=lam_node, fit_intercept=False, max_iter=10000)
            node_lasso.fit(X_nj, X_j)
            gamma_j = node_lasso.coef_
            r_j = X_j - X_nj @ gamma_j
            tau2_j = float(r_j @ X_j) / n
            if abs(tau2_j) < 1e-12:
                tau2_j = 1e-12  # guard against degenerate directions
            # Build the j-th row of Theta: insert -gamma_j with 1 at position j
            row = np.zeros(p)
            row[others] = -gamma_j
            row[j] = 1.0
            Theta[j, :] = row / tau2_j
            tau2[j] = tau2_j

        # ── Step 3: Debiased estimator ────────────────────────────────────────
        residual = y - X @ beta_hat
        b_hat = beta_hat + (1.0 / n) * Theta @ X.T @ residual

        # ── Step 4: Variance estimation ───────────────────────────────────────
        n_nonzero = np.sum(beta_hat != 0)
        dof = max(n - n_nonzero, 1)
        sigma2 = float(residual @ residual) / dof
        Omega = Theta @ (X.T @ X / n) @ Theta.T
        se = np.sqrt(sigma2 * np.diag(Omega) / n)

        return FitResult(
            beta_hat=b_hat,
            intercept=0.0,
            n_iter=None,
            diagnostics={
                "se": se,
                "z_stat": b_hat / np.where(se > 0, se, np.inf),
                "sigma2": sigma2,
                "beta_lasso": beta_hat,
            },
        )
