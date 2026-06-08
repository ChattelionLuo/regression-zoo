"""Group Lasso via block coordinate descent (card: ``group-lasso``).

Reference: Yuan & Lin (2006), *Model Selection and Estimation in Regression
with Grouped Variables*, Journal of the Royal Statistical Society B 68(1), 49-67.

Block coordinate update for group g (size d_g):
    rho_g = (1/n) X_g' (y - X_{-g} beta_{-g})
    beta_g = max(0, 1 - lam * sqrt(d_g) / ||rho_g||_2) * rho_g
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ...base import BaseSolver, FitResult
from ...links import Link

logger = logging.getLogger(__name__)

# card_id = "group-lasso"
# config: lam=0.1, groups=None, max_iter=1000, tol=1e-6


class GroupLassoSolver(BaseSolver):
    """Group Lasso via block coordinate descent.

    Config
    ------
    lam : float, default 0.1
    groups : list of lists of int, optional
        E.g. [[0,1,2],[3,4],[5]]. If None each feature is its own group
        (reduces to Lasso).
    max_iter : int, default 1000
    tol : float, default 1e-6
    fit_intercept : bool, default True
    """

    card_id = "group-lasso"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam: float = float(self.config.get("lam", 0.1))
        groups: Optional[List[List[int]]] = self.config.get("groups", None)
        max_iter: int = int(self.config.get("max_iter", 1000))
        tol: float = float(self.config.get("tol", 1e-6))
        fit_intercept: bool = bool(self.config.get("fit_intercept", True))

        if lnk.name != "identity":
            logger.warning(
                "GroupLassoSolver uses Gaussian fit for link=%r.", lnk.name
            )

        n, p = X.shape

        # centre
        if fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = float(y.mean())
            Xc = X - X_mean
            yc = y - y_mean
        else:
            Xc, yc = X, y
            X_mean = np.zeros(p)
            y_mean = 0.0

        if groups is None:
            groups = [[j] for j in range(p)]

        groups_arr = [np.asarray(g, dtype=int) for g in groups]

        beta = np.zeros(p)
        r = yc.copy().astype(float)
        converged = False

        for it in range(max_iter):
            beta_old = beta.copy()
            for g_idx in groups_arr:
                d_g = len(g_idx)
                Xg = Xc[:, g_idx]
                # add back current group contribution
                r_g = r + Xg @ beta[g_idx]
                rho_g = Xg.T @ r_g / n  # (d_g,)
                norm_rho = np.linalg.norm(rho_g)
                scale = max(0.0, 1.0 - lam * np.sqrt(d_g) / (norm_rho + 1e-14))
                beta[g_idx] = scale * rho_g
                r = r_g - Xg @ beta[g_idx]

            delta = np.max(np.abs(beta - beta_old))
            if delta < tol:
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
