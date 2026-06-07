"""glmzoo — faithful implementations of GLM solvers behind a common interface.

Phase 2 of the GLM Algorithm Arena. Each solver implements the contract::

    fit(X, y, link, **config) -> FitResult(beta_hat=..., ...)

so that, given identical (X, y, g), different solvers can be compared, correlated and
benchmarked across datasets in the Phase-3 arena.

This package currently provides the shared scaffolding (links, families, the base solver
interface). Concrete solvers are added alongside their documentation cards.
"""

from .base import BaseSolver, FitResult
from .links import LINKS, Link

__all__ = ["BaseSolver", "FitResult", "LINKS", "Link"]
__version__ = "0.0.1"
