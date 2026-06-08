"""Decorrelated Score test (card: ``decorrelated-score``).

Reference: Ning & Liu (2017). A general theory of hypothesis tests and confidence
regions for sparse high dimensional models. Ann. Statist. 45(1), 158-195.
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


class DecorrelatedScoreSolver(BaseSolver):
    """Decorrelated Score test for individual coordinates in high-dimensional GLMs.

    For each target coordinate j (or all if target_j is None) the score function
    is decorrelated with respect to nuisance coordinates via a nodewise Lasso.
    A Wald-type z-statistic is returned.

    Full inference is implemented for the identity (Gaussian) link; logistic
    inference uses the same decorrelation with the logistic score.

    Config
    ------
    lam      : float, default 0.1       — penalty for nuisance Lasso fit
    lam_w    : float, default 0.1       — penalty for decorrelation Lasso
    target_j : int | None, default None — coordinate(s) to test; None = all
    """

    card_id = "decorrelated-score"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        if not _HAS_SKLEARN:
            raise ImportError("sklearn is required for DecorrelatedScoreSolver")

        lnk = self._resolve_link(link)
        lam: float = self.config.get("lam", 0.1)
        lam_w: float = self.config.get("lam_w", 0.1)
        target_j = self.config.get("target_j", None)

        n, p = X.shape

        # ── Step 1: Fit nuisance Lasso ────────────────────────────────────────
        nuisance = _Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
        nuisance.fit(X, y)
        beta_hat = nuisance.coef_.copy()

        eta_hat = X @ beta_hat
        mu_hat = lnk.g_inv(eta_hat)
        V_hat = lnk.var(mu_hat)  # GLM variance V(mu)
        resid = y - mu_hat

        # Determine which coordinates to test
        if target_j is None:
            targets = list(range(p))
        elif isinstance(target_j, int):
            targets = [target_j]
        else:
            targets = list(target_j)

        cols = np.arange(p)
        psi = np.zeros(p)
        I_w = np.zeros(p)
        T_stat = np.zeros(p)

        for j in targets:
            others = cols[cols != j]
            X_j = X[:, j]
            X_oth = X[:, others]

            # Decorrelation: regress X_j on X_others via Lasso
            dec_lasso = _Lasso(alpha=lam_w, fit_intercept=False, max_iter=10000)
            dec_lasso.fit(X_oth, X_j)
            w_j = X_j - X_oth @ dec_lasso.coef_

            # Score statistic for j
            psi_j = float(w_j @ resid) / n
            I_w_j = float(w_j ** 2 @ V_hat) / n
            if I_w_j < 1e-12:
                I_w_j = 1e-12
            T_j = np.sqrt(n) * psi_j / np.sqrt(I_w_j)

            psi[j] = psi_j
            I_w[j] = I_w_j
            T_stat[j] = T_j

        return FitResult(
            beta_hat=beta_hat,
            intercept=0.0,
            n_iter=None,
            diagnostics={
                "psi": psi,
                "I_w": I_w,
                "T_stat": T_stat,
                "targets": targets,
                "beta_nuisance": beta_hat,
            },
        )
