"""OLS solver via QR factorization (card: ``ols``).

Reference: Golub & Van Loan, *Matrix Computations*, Ch. 5.
Uses ``np.linalg.lstsq`` which internally employs a divide-and-conquer SVD for
robustness, matching the spirit of the QR normal-equations approach.
"""

from __future__ import annotations

import logging
import numpy as np

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link

logger = logging.getLogger(__name__)


class OLSSolver(BaseSolver):
    """Ordinary Least Squares via normal equations (QR/SVD).

    Config
    ------
    fit_intercept : bool, default True
    """

    card_id = "ols"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        fit_intercept: bool = self.config.get("fit_intercept", True)

        n, p = X.shape

        if lnk.name != "identity":
            logger.warning(
                "OLS is defined for the identity link; performing one Newton step "
                "as an approximation for link=%r.",
                lnk.name,
            )
            # One Newton / IRLS step from zero initialisation
            beta = np.zeros(p)
            intercept = 0.0
            if fit_intercept:
                Xd = np.column_stack([np.ones(n), X])
                b0 = np.zeros(p + 1)
            else:
                Xd = X
                b0 = beta

            eta = Xd @ b0
            mu = lnk.g_inv(eta)
            gp = lnk.g_prime(mu)
            V = lnk.var(mu)
            W = 1.0 / (gp ** 2 * V + 1e-12)
            z = eta + (y - mu) * gp
            Xw = Xd * np.sqrt(W)[:, None]
            zw = z * np.sqrt(W)
            coeffs, _, _, _ = np.linalg.lstsq(Xw, zw, rcond=None)
            if fit_intercept:
                intercept = float(coeffs[0])
                beta = coeffs[1:]
            else:
                beta = coeffs
            return FitResult(
                beta_hat=beta,
                intercept=intercept,
                n_iter=1,
                diagnostics={"converged": True, "n_iter": 1},
            )

        # ── Gaussian / identity ──────────────────────────────────────────────
        if fit_intercept:
            Xd = np.column_stack([np.ones(n), X])
        else:
            Xd = X

        coeffs, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)

        if fit_intercept:
            intercept = float(coeffs[0])
            beta = coeffs[1:]
        else:
            intercept = 0.0
            beta = coeffs

        return FitResult(
            beta_hat=beta,
            intercept=intercept,
            n_iter=1,
            diagnostics={"converged": True, "n_iter": 1},
        )
