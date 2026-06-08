"""ISTA — Iterative Shrinkage-Thresholding Algorithm (card: ``ista``).

Reference: Beck & Teboulle (2009), *A Fast Iterative Shrinkage-Thresholding
Algorithm for Linear Inverse Problems*, SIAM J. Imaging Sciences 2(1), 183-202.

Algorithm (proximal gradient for Lasso)
-----------------------------------------
L  = max eigenvalue of X'X / n   (Lipschitz constant of the smooth part)
Repeat until ||beta_new - beta||_2 < tol:
    grad = -(1/n) X' (y - mu)          where mu = g_inv(X @ beta)
    beta_new = soft_threshold(beta - grad/L,  lam/L)
"""

from __future__ import annotations

import numpy as np

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link


def _soft_threshold(v: np.ndarray, t: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


class ISTASolver(BaseSolver):
    """ISTA proximal gradient descent for Lasso-penalised GLMs.

    Config
    ------
    lam : float, default 0.1
    max_iter : int, default 1000
    tol : float, default 1e-6
    backtracking : bool, default True
        Use Armijo backtracking line-search to adapt the step size.
    """

    card_id = "ista"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam: float = float(self.config.get("lam", 0.1))
        max_iter: int = int(self.config.get("max_iter", 1000))
        tol: float = float(self.config.get("tol", 1e-6))
        backtracking: bool = bool(self.config.get("backtracking", True))

        n, p = X.shape
        beta = np.zeros(p)

        # Lipschitz constant L = sigma_max(X'X) / n
        XtX = X.T @ X / n
        eigvals = np.linalg.eigvalsh(XtX)
        L = float(eigvals.max())
        if L < 1e-12:
            L = 1e-12

        converged = False
        for i in range(max_iter):
            eta = X @ beta
            mu = lnk.g_inv(eta)
            grad = -(X.T @ (y - mu)) / n

            if backtracking:
                # Simple Armijo backtracking
                step = 1.0 / L
                beta_new = _soft_threshold(beta - step * grad, lam * step)
                # Sufficient decrease check on the smooth part (Gaussian surrogate)
                for _ in range(50):
                    eta_new = X @ beta_new
                    mu_new = lnk.g_inv(eta_new)
                    f_new = 0.5 * np.mean((y - mu_new) ** 2)
                    f_old = 0.5 * np.mean((y - mu) ** 2)
                    quad = f_old + grad @ (beta_new - beta) + (L / 2) * np.sum((beta_new - beta) ** 2)
                    if f_new <= quad + 1e-12:
                        break
                    step *= 0.5
                    beta_new = _soft_threshold(beta - step * grad, lam * step)
            else:
                beta_new = _soft_threshold(beta - grad / L, lam / L)

            if np.linalg.norm(beta_new - beta) < tol:
                beta = beta_new
                converged = True
                break
            beta = beta_new

        n_iter = i + 1
        return FitResult(
            beta_hat=beta,
            intercept=0.0,
            n_iter=n_iter,
            diagnostics={"converged": converged, "n_iter": n_iter},
        )
