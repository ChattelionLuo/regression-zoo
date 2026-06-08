"""Ridge regression closed-form solver (card: ``ridge``).

Reference: Hoerl & Kennard (1970), *Ridge Regression: Biased Estimation for
Nonorthogonal Problems*, Technometrics 12(1).

Uses ``sklearn.linear_model.Ridge`` internally for the identity (Gaussian) link.
For non-identity links falls back to Gaussian with a warning.
"""

from __future__ import annotations

import logging
import numpy as np
from sklearn.linear_model import Ridge

from ...base import BaseSolver, FitResult
from ...links import LINKS, Link

logger = logging.getLogger(__name__)


class RidgeSolver(BaseSolver):
    """Ridge regression: beta = (X'X + lam*I)^{-1} X'y.

    Config
    ------
    lam : float, default 1.0
        L2 regularisation strength (alpha in sklearn).
    fit_intercept : bool, default True
    """

    card_id = "ridge"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam: float = float(self.config.get("lam", 1.0))
        fit_intercept: bool = self.config.get("fit_intercept", True)

        if lnk.name != "identity":
            logger.warning(
                "RidgeSolver closed form is for the identity link; "
                "falling back to Gaussian (identity) fit for link=%r.",
                lnk.name,
            )

        model = Ridge(alpha=lam, fit_intercept=fit_intercept)
        model.fit(X, y)

        beta = model.coef_.ravel()
        intercept = float(model.intercept_) if fit_intercept else 0.0

        return FitResult(
            beta_hat=beta,
            intercept=intercept,
            n_iter=1,
            diagnostics={"converged": True, "n_iter": 1},
        )
