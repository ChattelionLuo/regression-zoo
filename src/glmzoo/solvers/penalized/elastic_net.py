"""Elastic net solver (card: ``elastic-net``).

Reference: Zou & Hastie (2005), *Regularization and Variable Selection via the
Elastic Net*, Journal of the Royal Statistical Society B 67(2), 301-320.

Uses ``sklearn.linear_model.ElasticNet`` for the identity link and
``LogisticRegression(penalty='elasticnet', solver='saga')`` for the logit link.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import ElasticNet, LogisticRegression

from ...base import BaseSolver, FitResult
from ...links import Link

logger = logging.getLogger(__name__)

# card_id = "elastic-net"
# config: lam=0.1, alpha=0.5, max_iter=10000
# sklearn convention: ElasticNet(alpha=lam, l1_ratio=alpha_mix)


class ElasticNetSolver(BaseSolver):
    """Elastic net: combines L1 and L2 penalties.

    Config
    ------
    lam : float, default 0.1
        Total regularisation strength (sklearn ``alpha``).
    alpha : float, default 0.5
        Mixing parameter in [0, 1]; 1 = pure Lasso, 0 = pure Ridge
        (sklearn ``l1_ratio``).
    max_iter : int, default 10000
    fit_intercept : bool, default True
    """

    card_id = "elastic-net"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam: float = float(self.config.get("lam", 0.1))
        alpha: float = float(self.config.get("alpha", 0.5))
        max_iter: int = int(self.config.get("max_iter", 10000))
        fit_intercept: bool = bool(self.config.get("fit_intercept", True))

        if lnk.name == "logit":
            return self._fit_logit(X, y, lam, alpha, max_iter, fit_intercept)

        if lnk.name != "identity":
            logger.warning(
                "ElasticNetSolver uses Gaussian fit for link=%r.", lnk.name
            )

        model = ElasticNet(
            alpha=lam,
            l1_ratio=alpha,
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

    def _fit_logit(self, X, y, lam, alpha, max_iter, fit_intercept):
        n = X.shape[0]
        C = 1.0 / max(n * lam, 1e-12)
        model = LogisticRegression(
            penalty="elasticnet",
            C=C,
            solver="saga",
            l1_ratio=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
        )
        model.fit(X, y)
        beta = model.coef_.ravel()
        intercept = float(model.intercept_[0]) if fit_intercept else 0.0
        return FitResult(
            beta_hat=beta,
            intercept=intercept,
            n_iter=int(model.n_iter_[0]),
            diagnostics={"converged": int(model.n_iter_[0]) < max_iter, "n_iter": int(model.n_iter_[0])},
        )
