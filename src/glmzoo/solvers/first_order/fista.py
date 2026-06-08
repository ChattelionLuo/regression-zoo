"""FISTA — Fast ISTA with Nesterov momentum (card: ``fista``).

Reference: Beck & Teboulle (2009), *A Fast Iterative Shrinkage-Thresholding
Algorithm for Linear Inverse Problems*, SIAM J. Imaging Sciences 2(1), 183-202.

Algorithm
---------
Initialise y_k = beta = 0, t_k = 1
Repeat until ||beta_new - beta||_2 < tol:
    grad  = -(1/n) X' (y_obs - mu(y_k))
    beta_new = prox_{lam/L}(y_k - grad/L)
    t_new = (1 + sqrt(1 + 4*t_k^2)) / 2
    y_new = beta_new + ((t_k - 1) / t_new) * (beta_new - beta)
    beta, t_k, y_k = beta_new, t_new, y_new
"""

from __future__ import annotations

import numpy as np

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link


def _soft_threshold(v: np.ndarray, t: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


class FISTASolver(BaseSolver):
    """FISTA proximal gradient with Nesterov momentum for Lasso-penalised GLMs.

    Config
    ------
    lam : float, default 0.1
    max_iter : int, default 1000
    tol : float, default 1e-6
    """

    card_id = "fista"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam: float = float(self.config.get("lam", 0.1))
        max_iter: int = int(self.config.get("max_iter", 1000))
        tol: float = float(self.config.get("tol", 1e-6))

        n, p = X.shape
        beta = np.zeros(p)
        y_k = beta.copy()
        t_k = 1.0

        # Lipschitz constant
        XtX = X.T @ X / n
        eigvals = np.linalg.eigvalsh(XtX)
        L = float(eigvals.max())
        if L < 1e-12:
            L = 1e-12

        converged = False
        for i in range(max_iter):
            eta = X @ y_k
            mu = lnk.g_inv(eta)
            grad = -(X.T @ (y - mu)) / n

            beta_new = _soft_threshold(y_k - grad / L, lam / L)

            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
            y_new = beta_new + ((t_k - 1.0) / t_new) * (beta_new - beta)

            if np.linalg.norm(beta_new - beta) < tol:
                beta = beta_new
                converged = True
                break

            beta = beta_new
            t_k = t_new
            y_k = y_new

        n_iter = i + 1
        return FitResult(
            beta_hat=beta,
            intercept=0.0,
            n_iter=n_iter,
            diagnostics={"converged": converged, "n_iter": n_iter},
        )
