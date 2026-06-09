"""Lasso via coordinate descent (card: ``lasso-cd``).

Reference: Friedman, Hastie & Tibshirani (2007), *Pathwise Coordinate Optimization*,
Annals of Applied Statistics 1(2), 302-332.

Uses ``sklearn.linear_model.Lasso`` for the identity (Gaussian) link,
``LogisticRegression(penalty='l1', solver='saga')`` for the logit link,
and ``sklearn.linear_model.PoissonRegressor`` for the log link.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression, PoissonRegressor

from ...base import BaseSolver, FitResult
from ...links import Link

logger = logging.getLogger(__name__)

# card_id = "lasso-cd"
# config: lam=0.1, n_alphas=50, lam_min_ratio=0.01, max_iter=10000


class LassoCDSolver(BaseSolver):
    """Lasso via coordinate descent (glmnet-style path).

    Config
    ------
    lam : float, default 0.1
        Regularisation strength at which to report beta_hat.
    n_alphas : int, default 50
        Number of lambda values on the path.
    lam_min_ratio : float, default 0.01
        Ratio lam_min / lam_max.
    max_iter : int, default 10000
    fit_intercept : bool, default True
    """

    card_id = "lasso-cd"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam: float = float(self.config.get("lam", 0.1))
        n_alphas: int = int(self.config.get("n_alphas", 50))
        lam_min_ratio: float = float(self.config.get("lam_min_ratio", 0.01))
        max_iter: int = int(self.config.get("max_iter", 10000))
        fit_intercept: bool = bool(self.config.get("fit_intercept", True))

        n, p = X.shape

        if lnk.name == "logit":
            return self._fit_logit(X, y, lam, n_alphas, lam_min_ratio, max_iter, fit_intercept)
        if lnk.name == "log":
            return self._fit_poisson(X, y, lam, max_iter, fit_intercept)

        # --- Gaussian / identity link: fit directly at the requested lambda ---
        # (Previously used an internal path with lam_min_ratio clipping, which caused
        # many configs to produce identical betas.  Direct fit gives each lam its own
        # unique solution.)
        model = Lasso(alpha=lam, fit_intercept=fit_intercept, max_iter=max_iter)
        model.fit(X, y)

        return FitResult(
            beta_hat=model.coef_.ravel(),
            intercept=float(model.intercept_) if fit_intercept else 0.0,
            n_iter=model.n_iter_,
            diagnostics={"converged": model.n_iter_ < max_iter, "n_iter": model.n_iter_},
        )

    # ------------------------------------------------------------------
    def _fit_logit(self, X, y, lam, n_alphas, lam_min_ratio, max_iter, fit_intercept):
        n = X.shape[0]
        # sklearn LogisticRegression uses C = 1/(n*lam) convention
        C = 1.0 / max(n * lam, 1e-12)
        model = LogisticRegression(
            penalty="l1",
            C=C,
            solver="saga",
            fit_intercept=fit_intercept,
            max_iter=max_iter,
        )
        model.fit(X, y)
        beta = model.coef_.ravel()
        intercept = float(model.intercept_[0]) if fit_intercept else 0.0
        return FitResult(
            beta_hat=beta,
            intercept=intercept,
            n_iter=model.n_iter_[0],
            diagnostics={"converged": model.n_iter_[0] < max_iter, "n_iter": int(model.n_iter_[0])},
        )

    def _fit_poisson(self, X, y, lam, max_iter, fit_intercept):
        model = PoissonRegressor(
            alpha=lam,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
        )
        model.fit(X, y)
        return FitResult(
            beta_hat=model.coef_.ravel(),
            intercept=float(model.intercept_) if fit_intercept else 0.0,
            n_iter=model.n_iter_,
            diagnostics={"converged": model.n_iter_ < max_iter, "n_iter": model.n_iter_},
        )
