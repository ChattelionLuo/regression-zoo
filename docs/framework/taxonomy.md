# Taxonomy of GLM Algorithms

Algorithms are organized along three orthogonal axes. A single card may belong to several
categories; the **family** tag is its primary placement, the **data regime** and
**penalty / structure** tags refine it.

## Axis A — Algorithm family (how $\widehat\beta$ is computed)

| Family tag | Description | Representative methods |
|---|---|---|
| `classical` | closed-form / likelihood estimators for low-dim regression & GLMs | OLS, WLS, GLS, Ridge, IRLS / Fisher scoring, GLM-MLE |
| `penalized-batch` | full-data penalized likelihood with sparsity / structure | Lasso (CD), Elastic Net, Adaptive Lasso, SCAD, MCP, Group/Fused Lasso |
| `path-homotopy` | trace the exact solution path over $\lambda$ | LARS, Lasso homotopy, PDAS |
| `first-order-prox` | proximal / gradient solvers for the template objective | ISTA, FISTA, proximal-Newton, ADMM, coordinate descent |
| `high-dim-inference` | debiasing / corrected estimators that enable CIs & tests | debiased/desparsified Lasso, decorrelated score, post-double selection |
| `online-streaming` | sequential point/batch updates with bounded memory | SGD, implicit SGD, AdaGrad, FOBOS, RDA, truncated gradient, renewable estimation |
| `nonconvex-m` | nonconvex penalized M-estimation & local-optima theory | regularized nonconvex M-estimators |
| `estimating-eq` | defined via estimating equations rather than a loss | GEE, quasi-likelihood, QIF |

## Axis B — Data regime

| Regime tag | Condition | Typical assumptions |
|---|---|---|
| `low-dim` | $n \gg p$ | full-rank design, classical asymptotics |
| `high-dim` | $p \gtrsim n$ or $p \gg n$ | sparsity, restricted eigenvalue / compatibility |
| `streaming` | sequential arrival | bounded per-step compute & storage |

## Axis C — Penalty / structure

`none` · `ridge` · `lasso` · `elastic-net` · `adaptive-lasso` · `scad` · `mcp` ·
`group-lasso` · `fused-lasso` · `nonconvex` · `other`.

## How to read a card

Each card header shows badges for **family**, **regime**, and **status**, followed by a
metadata block, then the precise mathematics. See the [algorithm card format](algorithm-card.md).

## Coverage map (living)

The catalogue groups cards by family. As the encyclopedia grows, this taxonomy is the index
that keeps hundreds of solvers navigable and comparable. The future **arena** will use the
machine-readable tags (in each card's frontmatter and the aggregated
`registry/algorithms.yaml`) to build leaderboards and agreement analyses.
