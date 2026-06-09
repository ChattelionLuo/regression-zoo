"""Base solver interface for the GLM Algorithm Arena (Phase 2).

Every concrete solver subclasses ``BaseSolver`` and implements ``fit``, returning a
``FitResult``. The uniform contract is what makes cross-algorithm comparison possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .links import LINKS, Link


@dataclass
class FitResult:
    """Output of a solver: the estimate plus optional path / diagnostics."""

    beta_hat: np.ndarray
    intercept: float = 0.0
    n_iter: Optional[int] = None
    path: Optional[Any] = None            # e.g. coefficients over a lambda grid
    diagnostics: dict = field(default_factory=dict)


class BaseSolver:
    """Common interface: ``fit(X, y, link, **config) -> FitResult``.

    Subclasses should be *faithful* to the source algorithm documented in the matching
    card under docs/algorithms/. Keep configuration names aligned with the card's
    "Hyperparameters & configuration" section.
    """

    #: matches the card `id` in docs/algorithms/<id>.md
    card_id: str = "base"

    def __init__(self, config: dict | None = None, **kwargs: Any) -> None:
        # Accept either SolverCls(config={...}) or SolverCls(lam=0.1, ...)
        if config is not None:
            self.config = dict(config)
        else:
            self.config = kwargs

    def _resolve_link(self, link: str | Link) -> Link:
        if isinstance(link, Link):
            return link
        if link not in LINKS:
            raise ValueError(f"unknown link {link!r}; available: {sorted(LINKS)}")
        return LINKS[link]

    def fit(self, X: np.ndarray, y: np.ndarray, link: str | Link = "identity") -> FitResult:
        raise NotImplementedError
