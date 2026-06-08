"""Adaptive Lasso solver (card: ``adaptive-lasso``).

Reference: Zou (2006), *The Adaptive Lasso and Its Oracle Properties*,
Journal of the American Statistical Association 101(476), 1418-1429.

Two-stage procedure:
  Stage 1 — fit Ridge to obtain initial estimates beta0.
  Stage 2 — compute adaptive weights w_j = 1 / (|beta0_j| + eps)^gamma,
             rescale columns X_tilde = X / w_j, fit Lasso on (X_tilde, y),
             then unscale: beta_hat = lasso_coef / w_j.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import Lasso, Ridge

from ...base import BaseSolver, FitResult
from ...links import Link

logger = logging.getLogger(__name__)

# card_id = "adaptive-lasso"
# config: lam=0.1, gamma=1.0, lam_init=0.1, eps=1e-6


class AdaptiveLassoSolver(BaseSolver):
    """Adaptive Lasso via two-stage Ridge + weighted Lasso.

    Config
    ------
    lam : float, default 0.1
        Lasso penalty in Stage 2.
    gamma : float, default 1.0
        Exponent for computing adaptive weights.
    lam_init : float, default 0.1
        Ridge penalty in Stage 1.
    eps : float, default 1e-6
        Stabiliser in denominator of weights.
    max_iter : int, default 10000
    fit_intercept : bool, default True
    """

    card_id = "adaptive-lasso"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam: float = float(self.config.get("lam", 0.1))
        gamma: float = float(self.config.get("gamma", 1.0))
        lam_init: float = float(self.config.get("lam_init", 0.1))
        eps: float = float(self.config.get("eps", 1e-6))
        max_iter: int = int(self.config.get("max_iter", 10000))
        fit_intercept: bool = bool(self.config.get("fit_intercept", True))

        if lnk.name != "identity":
            logger.warning(
                "AdaptiveLassoSolver is designed for the identity link; "
                "proceeding with Gaussian fit for link=%r.",
                lnk.name,
            )

        # Stage 1: Ridge
        ridge = Ridge(alpha=lam_init, fit_intercept=fit_intercept)
        ridge.fit(X, y)
        beta0 = ridge.coef_.ravel()

        # Stage 2: adaptive weights and rescaled Lasso
        weights = 1.0 / (np.abs(beta0) + eps) ** gamma
        X_tilde = X / weights[np.newaxis, :]

        lasso = Lasso(alpha=lam, fit_intercept=fit_intercept, max_iter=max_iter)
        lasso.fit(X_tilde, y)

        beta_tilde = lasso.coef_.ravel()
        beta_hat = beta_tilde / weights

        return FitResult(
            beta_hat=beta_hat,
            intercept=float(lasso.intercept_) if fit_intercept else 0.0,
            n_iter=lasso.n_iter_,
            diagnostics={"converged": lasso.n_iter_ < max_iter, "n_iter": lasso.n_iter_},
        )
