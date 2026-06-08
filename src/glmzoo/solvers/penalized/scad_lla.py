"""SCAD via Local Linear Approximation (card: ``scad-lla``).

Reference: Zou & Li (2008), *One-Step Sparse Estimates in Nonconcave Penalized
Likelihood Models*, Annals of Statistics 36(4), 1509-1533.

The LLA algorithm linearises the SCAD penalty around the current iterate,
yielding a sequence of weighted L1 (Lasso) problems solved to convergence.

SCAD penalty derivative (Fan & Li 2001):
    p'(t; lam, a) = lam * I(t <= lam)
                   + (a*lam - t) / ((a-1)*lam) * I(lam < t <= a*lam)
                   + 0                           * I(t > a*lam)
with a = 3.7.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import Lasso

from ...base import BaseSolver, FitResult
from ...links import Link

logger = logging.getLogger(__name__)

# card_id = "scad-lla"
# config: lam=0.1, a=3.7, max_iter=50, tol=1e-6, eps=1e-8


def _scad_deriv(t: np.ndarray, lam: float, a: float) -> np.ndarray:
    """Element-wise SCAD penalty derivative evaluated at |t|."""
    t = np.abs(t)
    d = np.where(
        t <= lam,
        lam,
        np.where(
            t <= a * lam,
            (a * lam - t) / ((a - 1) * lam),
            0.0,
        ),
    )
    return d


class SCADLLASolver(BaseSolver):
    """SCAD via Local Linear Approximation (LLA).

    Config
    ------
    lam : float, default 0.1
    a : float, default 3.7
        SCAD shape parameter.
    max_iter : int, default 50
        LLA outer iterations.
    tol : float, default 1e-6
        Convergence tolerance on beta.
    eps : float, default 1e-8
        Stabiliser in weight denominator.
    max_iter_lasso : int, default 10000
        sklearn Lasso inner iterations.
    fit_intercept : bool, default True
    """

    card_id = "scad-lla"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam: float = float(self.config.get("lam", 0.1))
        a: float = float(self.config.get("a", 3.7))
        max_iter: int = int(self.config.get("max_iter", 50))
        tol: float = float(self.config.get("tol", 1e-6))
        eps: float = float(self.config.get("eps", 1e-8))
        max_iter_lasso: int = int(self.config.get("max_iter_lasso", 10000))
        fit_intercept: bool = bool(self.config.get("fit_intercept", True))

        if lnk.name != "identity":
            logger.warning(
                "SCADLLASolver uses Gaussian fit for link=%r.", lnk.name
            )

        n, p = X.shape
        beta = np.zeros(p)
        converged = False

        for it in range(max_iter):
            # LLA weights are p'_SCAD(|beta_j|) directly (not divided by |beta_j|)
            weights = _scad_deriv(beta, lam, a)
            # Avoid zero weights (unpenalised region): use tiny floor so rescaling is stable
            weights_safe = np.maximum(weights, eps)

            # Variable substitution: b_tilde_j = w_j * b_j  =>  X_tilde_j = X_j / w_j
            # Weighted-L1 problem becomes Lasso(alpha=1) on (X_tilde, y)
            X_tilde = X / weights_safe[np.newaxis, :]
            lasso = Lasso(alpha=1.0, fit_intercept=fit_intercept, max_iter=max_iter_lasso)
            lasso.fit(X_tilde, y)
            beta_new = lasso.coef_.ravel() / weights_safe

            delta = np.max(np.abs(beta_new - beta))
            beta = beta_new
            if delta < tol:
                converged = True
                break

        return FitResult(
            beta_hat=beta,
            intercept=float(lasso.intercept_) if fit_intercept else 0.0,
            n_iter=it + 1,
            diagnostics={"converged": converged, "n_iter": it + 1},
        )
