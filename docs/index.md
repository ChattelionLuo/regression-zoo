# GLM Algorithm Arena

A comprehensive, faithful **encyclopedia of concrete algorithms (solvers) for Generalized
Linear Models** — and, in later phases, an **arena** where those algorithms compete on many
datasets.

## The one question every card answers

> **Given** a design matrix $X$, a response $y$, and a link function $g$,
> **how** does this algorithm produce a coefficient estimate $\hat\beta$ —
> under what configuration and hyperparameters?

From ordinary least squares to penalized high-dimensional estimators, from offline coordinate
descent to online stochastic approximation, every method is reduced to the same essential
mapping $(X, y, g) \mapsto \hat\beta$ and documented under [one unified notation](framework/notation.md).

## How it is organized

- **[Notation & framework](framework/notation.md)** — the shared mathematical language:
  exponential family, links, loss, score/Fisher information, the penalized-likelihood
  template $\hat\beta=\arg\min \mathcal L(\beta)+\lambda P(\beta)$, and the optimization primitives.
- **[Taxonomy](framework/taxonomy.md)** — how algorithms are grouped (family × data regime × penalty).
- **[Card schema](framework/card-schema.md)** — the exact template each algorithm card follows.
- **[Algorithm catalogue](algorithms/index.md)** — the growing library of cards.

## Project phases

1. **Documentation (current).** Faithful algorithm cards — exact mathematics only.
2. **Implementation (later).** Python implementations behind a common
   `fit(X, y, link, **config) -> beta_hat` interface, mirroring each method's official construction.
3. **Arena / analysis (later).** Run all solvers across many datasets; study performance,
   agreement, and correlations between their outputs.

## Scope (Phase 1)

Concrete **solvers** only: the precise algorithm mapping $(X,y,g)$ to $\hat\beta$, with its
configuration. Theory (rates, inference validity) is summarized briefly and cited, not proved.

!!! tip "Contributing a card"
    Copy the stub in the [card schema](framework/card-schema.md), drop it in
    `docs/algorithms/<id>.md`, fill the fixed sections faithfully from the source paper, and
    add the bibkey to `reference.bib`. Then it appears in the catalogue.
