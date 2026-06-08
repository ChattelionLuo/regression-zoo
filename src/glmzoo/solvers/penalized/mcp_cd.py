"""MCP via coordinate descent (card: ``mcp-cd``).

Reference: Zhang (2010), *Nearly Unbiased Variable Selection Under Minimax
Concave Penalty*, Annals of Statistics 38(2), 894-942.

Coordinate descent with the MCP soft-threshold update:
    rho_j = (1/n) X_j' (y - X_{-j} beta_{-j})
    if   |rho_j| <= lam:          beta_j = 0
    elif |rho_j| <= gamma*lam:    beta_j = sign(rho_j)*(|rho_j|-lam)/(1-1/gamma)
    else:                          beta_j = rho_j
"""

from __future__ import annotations

import logging

import numpy as np

from ...base import BaseSolver, FitResult
from ...links import Link

logger = logging.getLogger(__name__)

# card_id = "mcp-cd"
# config: lam=0.1, gamma=3.0, max_iter=1000, tol=1e-6


def _mcp_threshold(rho: float, lam: float, gamma: float) -> float:
    """MCP coordinate update."""
    abs_rho = abs(rho)
    if abs_rho <= lam:
        return 0.0
    if abs_rho <= gamma * lam:
        return np.sign(rho) * (abs_rho - lam) / (1.0 - 1.0 / gamma)
    return rho


class MCPCDSolver(BaseSolver):
    """MCP via coordinate descent.

    Config
    ------
    lam : float, default 0.1
    gamma : float, default 3.0
        MCP shape parameter (> 1).
    max_iter : int, default 1000
    tol : float, default 1e-6
    fit_intercept : bool, default True
    """

    card_id = "mcp-cd"

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        lnk = self._resolve_link(link)
        lam: float = float(self.config.get("lam", 0.1))
        gamma: float = float(self.config.get("gamma", 3.0))
        max_iter: int = int(self.config.get("max_iter", 1000))
        tol: float = float(self.config.get("tol", 1e-6))
        fit_intercept: bool = bool(self.config.get("fit_intercept", True))

        if lnk.name != "identity":
            logger.warning(
                "MCPCDSolver uses Gaussian fit for link=%r.", lnk.name
            )

        n, p = X.shape
        # centre y and X if fitting intercept
        if fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = float(y.mean())
            Xc = X - X_mean
            yc = y - y_mean
        else:
            Xc, yc = X, y
            X_mean = np.zeros(p)
            y_mean = 0.0

        # precompute column norms squared
        col_norms_sq = np.sum(Xc ** 2, axis=0) / n  # (p,)

        beta = np.zeros(p)
        r = yc.copy().astype(float)  # residual
        converged = False

        for it in range(max_iter):
            beta_old = beta.copy()
            for j in range(p):
                # partial residual: add back contribution of j
                r_j = r + Xc[:, j] * beta[j]
                rho_j = float(Xc[:, j] @ r_j) / n
                # normalise by column norm
                nrm = col_norms_sq[j]
                if nrm < 1e-14:
                    beta[j] = 0.0
                    r = r_j
                    continue
                rho_j_scaled = rho_j / nrm
                lam_scaled = lam / nrm
                gamma_eff = gamma  # gamma * nrm / nrm cancels

                beta[j] = _mcp_threshold(rho_j_scaled, lam_scaled, gamma_eff)
                r = r_j - Xc[:, j] * beta[j]

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
