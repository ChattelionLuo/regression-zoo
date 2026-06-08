"""IRLS solver for generalised linear models (card: ``glm-irls``).

Reference: Green (1984), *Iteratively Reweighted Least Squares for Maximum
Likelihood Estimation*, JRSS-B 46(2), 149-192.

Algorithm
---------
Repeat until convergence:
    eta = X @ beta
    mu  = g_inv(eta)
    W   = diag(1 / (g'(mu)^2 * V(mu)))   working weights
    z   = eta + (y - mu) * g'(mu)         working response
    beta_new = (X'WX)^{-1} X'Wz          WLS step via lstsq
"""

from __future__ import annotations

import numpy as np

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link


class GLMIRLSSolver(BaseSolver):
    """Iteratively Reweighted Least Squares for GLMs.

    Config
    ------
    max_iter : int, default 100
    tol : float, default 1e-8
    fit_intercept : bool, default True
    """

    card_id = "glm-irls"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        max_iter: int = int(self.config.get("max_iter", 100))
        tol: float = float(self.config.get("tol", 1e-8))
        fit_intercept: bool = self.config.get("fit_intercept", True)

        n, p = X.shape
        if fit_intercept:
            Xd = np.column_stack([np.ones(n), X])
        else:
            Xd = X

        d = Xd.shape[1]
        beta = np.zeros(d)
        converged = False

        for i in range(max_iter):
            eta = Xd @ beta
            mu = lnk.g_inv(eta)
            gp = lnk.g_prime(mu)
            V = lnk.var(mu)
            W = 1.0 / (gp ** 2 * V + 1e-12)
            z = eta + (y - mu) * gp

            sqrtW = np.sqrt(W)
            Xw = Xd * sqrtW[:, None]
            zw = z * sqrtW

            beta_new, _, _, _ = np.linalg.lstsq(Xw, zw, rcond=None)

            if np.linalg.norm(beta_new - beta) < tol:
                beta = beta_new
                converged = True
                break
            beta = beta_new

        n_iter = i + 1

        if fit_intercept:
            intercept = float(beta[0])
            beta_hat = beta[1:]
        else:
            intercept = 0.0
            beta_hat = beta

        return FitResult(
            beta_hat=beta_hat,
            intercept=intercept,
            n_iter=n_iter,
            diagnostics={"converged": converged, "n_iter": n_iter},
        )
