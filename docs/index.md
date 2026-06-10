---
hide:
  - toc
---

<div class="arena-hero" markdown>

# An encyclopedia of GLM algorithms

<p class="arena-lede" markdown>
Linear and generalized linear models are still load-bearing statistical tools. Their algorithms are scattered across papers, packages, and systems. This project puts them in one coordinate system.
</p>

<div class="arena-cta" markdown>
[Browse the catalogue](algorithms/index.md){ .arena-btn .arena-btn--primary }
[Explore the Arena](arena/index.md){ .arena-btn .arena-btn--ghost }
</div>

</div>

<p class="arena-kicker">The one question every card answers</p>

> **Given** a design matrix $X$, a response vector $y$, a model specification — family, link $g$, loss, and tuning parameters where relevant —
> **what** exactly does the algorithm return as $\widehat\beta$ ?

Even someone in finance can parrot $\widehat\beta = (X^\top X)^{-1} X^\top y$, but it is much trickier when the situation becomes irregular. In statistics, each method can be essential viewed as the mapping $(X, y, \text{model}) \mapsto \widehat\beta$ and often written against the shared template

$$
\widehat\beta\in\arg\min_\beta \mathcal L(\beta; X, y, g)+\lambda P(\beta),
$$

or other procedures/updating rules. In here, estimators are defined across decades of literature that can be read, compared, and — in later phases — benchmarked on equal footing.

## Explore

<div class="grid cards" markdown>

-   :material-function-variant:{ .lg } &nbsp; **Notation & framework**

    ---

    The shared mathematical language: exponential family, link functions, loss,
    score and Fisher information, penalties, and the optimization primitives every card builds on.

    [:octicons-arrow-right-24: Read the framework](framework/notation.md)

-   :material-card-text-outline:{ .lg } &nbsp; **Algorithm card**

    ---

    The template that keeps the catalogue honest: metadata, source-backed mathematics, and a rule against undocumented reinterpretations.

    [:octicons-arrow-right-24: View the algorithm card format](framework/algorithm-card.md)

-   :material-bookshelf:{ .lg } &nbsp; **Algorithm catalogue**

    ---

    The growing collection: OLS, Ridge, Lasso, Elastic Net, SCAD, MCP, LARS, FISTA,
    debiased Lasso, SGD, and many more.

    [:octicons-arrow-right-24: Open the catalogue](algorithms/index.md)

-   :material-chart-scatter-plot:{ .lg } &nbsp; **The Arena**

    ---

    Every algorithm run across all datasets with multiple hyperparameter configs each.
    Interactive t-SNE / UMAP embeddings and coefficient heatmaps reveal how methods cluster and differ.

    [:octicons-arrow-right-24: Explore the Arena](arena/index.md)

</div>

## Why this project exists

Linear models are old in the way linear algebra is old: mature, not obsolete. They sit underneath econometric studies, experimental analysis, clinical and credit-risk scores, fraud systems, demand and traffic models, industrial monitoring, and scientific baselines. Sometimes they are the final model. Often they are the baseline. Almost always they are the thing you want to understand before trusting something larger.

The algorithms around them are less tidy. The same statistical object can appear under different notation; the same penalty can be fit by several solvers; likelihood, regularization, inference, and streaming updates often live in different literatures.

This Arena puts this solver zoo in one coordinate system. Each algorithm card states the estimator, assumptions, objective or update rule, hyperparameters, and output contract using shared notation and source-faithful mathematics.

## Project construction

<div class="grid cards" markdown>

-   **I · Documentation** &nbsp; <span class="badge status-current">current</span>

    ---

    Faithful algorithm cards — exact mathematics only. Each defines the estimator,
    the algorithm, its hyperparameters, and the precise $(X,y,g)\mapsto\widehat\beta$ contract.

-   **II · Implementation & Arena** &nbsp; <span class="badge status-current">current</span>

    ---

    Python solvers behind one interface, `fit(X, y, link, **config) -> beta_hat`,
    mirroring each method's official construction. Run across 16 datasets with 50+ configs per solver.
    [Explore the Arena →](arena/index.md)

-   **III · Deep analysis** &nbsp; <span class="badge status-stub">planned</span>

    ---

    Structured comparison of solver families: convergence rates, sensitivity to
    regularisation, agreement on held-out data, and statistical inference properties.

</div>

!!! tip "Contributing an algorithm card"
    If a source defines an algorithm precisely, it can become an algorithm card. See [algorithm card](framework/algorithm-card.md) for the template.
    We welcome contributions of new algorithm cards, datasets, or hyperparameter tuning suggestions.
    Feel free to reach out at <a href="mailto:chattelion.luo@connect.polyu.hk">chattelion.luo@connect.polyu.hk</a>.
