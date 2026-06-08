"""Fused Lasso via ADMM (card: ``fused-lasso``).

Reference: Tibshirani et al. (2005), *Sparsity and Smoothness via the Fused
Lasso*, Journal of the Royal Statistical Society B 67(1), 91-108.
ADMM formulation follows Boyd et al. (2011), *Distributed Optimization and
Statistical Learning via the Alternating Direction Method of Multipliers*.

Solves:
    min  (1/2n)||y - X beta||^2 + lam1*||beta||_1 + lam2*sum_j|beta_j - beta_{j-1}|

ADMM splitting with z = [beta; D beta] (D = first-difference matrix):
    beta-update:  (X'X/n + rho*(I + D'D)) beta = X'y/n + rho*(z_lasso - u_lasso + D'(z_fused - u_fused))
    z-update:     z_lasso  = soft_thresh(beta + u_lasso,  lam1/rho)
                  z_fused  = soft_thresh(D beta + u_fused, lam2/rho)
    dual-update:  u_lasso  += beta  - z_lasso
                  u_fused  += D beta - z_fused
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import linalg

from ...base import BaseSolver, FitResult
from ...links import Link

logger = logging.getLogger(__name__)

# card_id = "fused-lasso"
# config: lam1=0.1, lam2=0.1, rho=1.0, max_iter=1000, tol=1e-4


def _soft_thresh(v: np.ndarray, t: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def _diff_matrix(p: int) -> np.ndarray:
    """First-difference matrix D of shape (p-1, p)."""
    D = np.zeros((p - 1, p))
    for i in range(p - 1):
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    return D


class FusedLassoSolver(BaseSolver):
    """Fused Lasso via ADMM (identity link only).

    Config
    ------
    lam1 : float, default 0.1
        Sparsity (L1) penalty.
    lam2 : float, default 0.1
        Fusion (total variation) penalty.
    rho : float, default 1.0
        ADMM step size.
    max_iter : int, default 1000
    tol : float, default 1e-4
        Primal and dual residual tolerance.
    fit_intercept : bool, default True
    """

    card_id = "fused-lasso"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam1: float = float(self.config.get("lam1", 0.1))
        lam2: float = float(self.config.get("lam2", 0.1))
        rho: float = float(self.config.get("rho", 1.0))
        max_iter: int = int(self.config.get("max_iter", 1000))
        tol: float = float(self.config.get("tol", 1e-4))
        fit_intercept: bool = bool(self.config.get("fit_intercept", True))

        if lnk.name != "identity":
            logger.warning(
                "FusedLassoSolver is designed for the identity link; "
                "proceeding with Gaussian fit for link=%r.",
                lnk.name,
            )

        n, p = X.shape

        if fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = float(y.mean())
            Xc = X - X_mean
            yc = y - y_mean
        else:
            Xc, yc = X, y
            X_mean = np.zeros(p)
            y_mean = 0.0

        XtX = Xc.T @ Xc / n
        Xty = Xc.T @ yc / n

        D = _diff_matrix(p)           # (p-1, p)
        DtD = D.T @ D                 # (p, p)

        # Precompute and factor A = X'X/n + rho*(I + D'D)
        A = XtX + rho * (np.eye(p) + DtD)
        try:
            L = linalg.cholesky(A, lower=True)
            use_chol = True
        except linalg.LinAlgError:
            use_chol = False

        def _solve(rhs):
            if use_chol:
                return linalg.cho_solve((L, True), rhs)
            return linalg.solve(A, rhs)

        beta = np.zeros(p)
        z_lasso = np.zeros(p)
        z_fused = np.zeros(p - 1)
        u_lasso = np.zeros(p)
        u_fused = np.zeros(p - 1)

        converged = False
        for it in range(max_iter):
            # beta update
            rhs = Xty + rho * (z_lasso - u_lasso) + rho * D.T @ (z_fused - u_fused)
            beta_new = _solve(rhs)

            # z updates
            z_lasso_new = _soft_thresh(beta_new + u_lasso, lam1 / rho)
            z_fused_new = _soft_thresh(D @ beta_new + u_fused, lam2 / rho)

            # dual updates
            u_lasso = u_lasso + beta_new - z_lasso_new
            u_fused = u_fused + D @ beta_new - z_fused_new

            # residuals
            primal_res = np.linalg.norm(np.concatenate([
                beta_new - z_lasso_new,
                D @ beta_new - z_fused_new,
            ]))
            dual_res = rho * np.linalg.norm(np.concatenate([
                z_lasso_new - z_lasso,
                z_fused_new - z_fused,
            ]))

            beta = beta_new
            z_lasso = z_lasso_new
            z_fused = z_fused_new

            if primal_res < tol and dual_res < tol:
                converged = True
                break

        intercept = 0.0
        if fit_intercept:
            intercept = y_mean - float(X_mean @ beta)

        return FitResult(
            beta_hat=beta,
            intercept=intercept,
            n_iter=it + 1,
            diagnostics={"converged": converged, "n_iter": it + 1},
        )
