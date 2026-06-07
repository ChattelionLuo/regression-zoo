---
hide:
  - toc
---

<div class="arena-hero" markdown>

<span class="arena-eyebrow">GLM Algorithm Arena · Phase I</span>

# An encyclopedia of GLM algorithms.

<p class="arena-lede" markdown>
Linear and generalized linear models are still load-bearing statistical tools. Their algorithms are scattered across papers, packages, and notation systems. This project puts them in one coordinate system.
</p>

<div class="arena-cta" markdown>
[Browse the catalogue](algorithms/index.md){ .arena-btn .arena-btn--primary }
[Read the framework](framework/notation.md){ .arena-btn .arena-btn--ghost }
</div>

</div>

<p class="arena-kicker">The one question every card answers</p>

> **Given** a design matrix $X$, a response vector $y$, a model specification — family, link $g$, loss, and tuning parameters where relevant —
> **what** exactly does the algorithm return as $\hat\beta$ ?

Each method can be viewed as the same essential mapping $(X, y, \text{model}) \mapsto \hat\beta$ and often written
against the shared template 
$$\;\hat\beta\in\arg\min_\beta \mathcal L(\beta; X, y, g)+\lambda P(\beta)$$
, or other procedures/ output rules. In here, estimators are defined across decades of literature that can be read, compared, and — in later phases —
benchmarked on equal footing.

## Explore

<div class="grid cards" markdown>

-   :material-function-variant:{ .lg } &nbsp; **Notation & framework**

    ---

    The shared mathematical language: exponential family, link functions, loss,
    score and Fisher information, penalties, and the optimization primitives every card builds on.

    [:octicons-arrow-right-24: Read the framework](framework/notation.md)

-   :material-sitemap-outline:{ .lg } &nbsp; **Taxonomy**

    ---

    How hundreds of solvers stay navigable,
    from classical closed forms to online stochastic approximation.

    [:octicons-arrow-right-24: See the taxonomy](framework/taxonomy.md)

-   :material-card-text-outline:{ .lg } &nbsp; **Card schema**

    ---

    The template that keeps the catalogue honest: metadata, source-backed mathematics, and a rule against undocumented reinterpretations.

    [:octicons-arrow-right-24: View the schema](framework/card-schema.md)

-   :material-bookshelf:{ .lg } &nbsp; **Algorithm catalogue**

    ---

    The growing collection: OLS, Ridge, Lasso, Elastic Net, SCAD, MCP, LARS, FISTA,
    debiased Lasso, SGD, and many more.

    [:octicons-arrow-right-24: Open the catalogue](algorithms/index.md)

</div>

## Project construction

<div class="grid cards" markdown>

-   **I · Documentation** &nbsp; <span class="badge status-draft">current</span>

    ---

    Faithful algorithm cards — exact mathematics only. Each defines the estimator,
    the algorithm, its hyperparameters, and the precise $(X,y,g)\mapsto\hat\beta$ contract.

-   **II · Implementation** &nbsp; <span class="badge status-stub">planned</span>

    ---

    Python solvers behind one interface, `fit(X, y, link, **config) -> beta_hat`,
    mirroring each method's official construction.

-   **III · The Arena** &nbsp; <span class="badge status-stub">planned</span>

    ---

    Run every solver across many datasets; study performance, agreement, and the
    correlations between their outputs.

</div>

!!! tip "Contributing a card"
    If a source defines an algorithm precisely, it can become a card. Copy the stub in the [card schema](framework/card-schema.md), drop it in
    `docs/algorithms/<id>.md`, fill the fixed sections faithfully from the source paper, and
    add the bibkey to `reference.bib`. Run `python scripts/validate_cards.py`, and it appears
    in the catalogue.
