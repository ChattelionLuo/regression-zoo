"""Link functions and exponential-family variance functions.

Notation follows docs/framework/notation.md:
    eta = X @ beta ,  mu = g_inv(eta) ,  g(mu) = eta .
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Link:
    """A link g with its inverse, derivative, and the family variance function V(mu)."""

    name: str
    g: Callable[[np.ndarray], np.ndarray]          # mu  -> eta
    g_inv: Callable[[np.ndarray], np.ndarray]      # eta -> mu
    g_prime: Callable[[np.ndarray], np.ndarray]    # d eta / d mu = g'(mu)
    var: Callable[[np.ndarray], np.ndarray]        # V(mu), the family variance function


_EPS = 1e-12


identity = Link(
    name="identity",
    g=lambda mu: mu,
    g_inv=lambda eta: eta,
    g_prime=lambda mu: np.ones_like(mu),
    var=lambda mu: np.ones_like(mu),
)

logit = Link(
    name="logit",
    g=lambda mu: np.log(mu / (1.0 - mu)),
    g_inv=lambda eta: 1.0 / (1.0 + np.exp(-eta)),
    g_prime=lambda mu: 1.0 / (mu * (1.0 - mu) + _EPS),
    var=lambda mu: mu * (1.0 - mu),
)

log = Link(
    name="log",
    g=lambda mu: np.log(mu + _EPS),
    g_inv=lambda eta: np.exp(eta),
    g_prime=lambda mu: 1.0 / (mu + _EPS),
    var=lambda mu: mu,
)

LINKS = {l.name: l for l in (identity, logit, log)}
