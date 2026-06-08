"""LARS / LARS-Lasso path solver (card: ``lars``).

Reference: Efron, Hastie, Johnstone & Tibshirani (2004), *Least Angle Regression*,
Annals of Statistics 32(2), 407-499.

Delegates to ``sklearn.linear_model.lars_path`` and exposes the full coefficient
path via ``FitResult.path``.  Only the identity (Gaussian / LS) link is supported;
for other links a ``NotImplementedError`` is raised.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import lars_path

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link


class LARSSolver(BaseSolver):
    """LARS / LARS-Lasso path (identity link only).

    Config
    ------
    lam : float or None, default None
        Target regularisation strength.  If None the coefficient at the median
        alpha along the path is returned.
    n_alphas : int, default 100
        Maximum number of path steps passed to ``lars_path``.
    max_iter : int, default 500
        Maximum iterations passed to ``lars_path``.
    method : str, default "lasso"
        ``"lasso"`` or ``"lar"``.
    """

    card_id = "lars"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        if lnk.name != "identity":
            raise NotImplementedError(
                f"LARSSolver supports the identity link only; got {lnk.name!r}."
            )

        lam = self.config.get("lam", None)
        max_iter: int = int(self.config.get("max_iter", 500))
        method: str = self.config.get("method", "lasso")

        alphas, _, coefs = lars_path(X, y, method=method, max_iter=max_iter)
        # coefs shape: (p, n_path_points)

        if lam is None:
            # Use the coefficient at the median alpha
            idx = len(alphas) // 2
        else:
            idx = int(np.argmin(np.abs(alphas - float(lam))))

        beta = coefs[:, idx]

        return FitResult(
            beta_hat=beta,
            intercept=0.0,
            n_iter=coefs.shape[1],
            path={"alphas": alphas, "coefs": coefs},
            diagnostics={"converged": True, "n_iter": coefs.shape[1]},
        )
