---
hide:
  - toc
---

<div class="arena-hero" markdown>

<span class="arena-eyebrow">GLM Algorithm Arena · Phase I</span>

# An encyclopedia of algorithms for generalized linear models.

<p class="arena-lede" markdown>
Every estimator for linear and generalized linear models — classical, penalized,
high-dimensional, and online — reduced to **one precise mapping** and documented under
**one unified notation**. From OLS to debiased Lasso to streaming stochastic approximation.
</p>

<div class="arena-cta" markdown>
[Browse the catalogue](algorithms/index.md){ .arena-btn .arena-btn--primary }
[Read the framework](framework/notation.md){ .arena-btn .arena-btn--ghost }
</div>

</div>

<p class="arena-kicker">The one question every card answers</p>

> **Given** a design matrix $X$, a response $y$, and a link function $g$,
> **how** does this algorithm produce a coefficient estimate $\hat\beta$ —
> under what configuration and hyperparameters?

Each method is reduced to the same essential mapping $(X, y, g) \mapsto \hat\beta$ and written
against the shared template $\;\hat\beta=\arg\min_\beta \mathcal L(\beta)+\lambda P(\beta)$, so
estimators defined across decades of literature can be read, compared, and — in later phases —
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

    How hundreds of solvers stay navigable — organized by **family × data regime × penalty**,
    from classical closed forms to online stochastic approximation.

    [:octicons-arrow-right-24: See the taxonomy](framework/taxonomy.md)

-   :material-card-text-outline:{ .lg } &nbsp; **Card schema**

    ---

    The exact, faithful template each algorithm follows — machine-readable metadata plus
    nine fixed sections, with a strict faithfulness rule.

    [:octicons-arrow-right-24: View the schema](framework/card-schema.md)

-   :material-bookshelf:{ .lg } &nbsp; **Algorithm catalogue**

    ---

    The growing library: OLS, Ridge, Lasso, Elastic Net, SCAD, MCP, LARS, FISTA,
    debiased Lasso, SGD, and many more.

    [:octicons-arrow-right-24: Open the catalogue](algorithms/index.md)

</div>

## Built in three phases

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
    Copy the stub in the [card schema](framework/card-schema.md), drop it in
    `docs/algorithms/<id>.md`, fill the fixed sections faithfully from the source paper, and
    add the bibkey to `reference.bib`. Run `python scripts/validate_cards.py`, and it appears
    in the catalogue.
