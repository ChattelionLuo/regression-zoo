<p class="arena-kicker">Phase I · Documentation</p>

# Algorithm Catalogue

The library of algorithm cards, grouped by [family](../framework/taxonomy.md). Each entry maps
$(X, y, g) \mapsto \hat\beta$ under the [unified notation](../framework/notation.md).

!!! note "Status legend"
    **reviewed** — math checked against source · **draft** — written, pending review ·
    **stub** — scaffold only.

## Classical (low-dimensional regression & GLMs)

- [Ordinary Least Squares (OLS)](ols.md)
- [Ridge Regression](ridge.md)
- [GLM via IRLS / Fisher Scoring](glm-irls.md)

## Penalized (batch)

- [Lasso via Coordinate Descent](lasso-cd.md)
- [Elastic Net](elastic-net.md)
- [Adaptive Lasso](adaptive-lasso.md)
- [SCAD via Local Linear Approximation](scad-lla.md)
- [MCP via Coordinate Descent](mcp-cd.md)
- [Group Lasso](group-lasso.md)
- [Fused Lasso](fused-lasso.md)

## Path / homotopy

- [Least Angle Regression (LARS)](lars.md)

## First-order / proximal solvers

- [ISTA](ista.md)
- [FISTA](fista.md)

## High-dimensional inference

- [Debiased / Desparsified Lasso](debiased-lasso.md)
- [Decorrelated Score (Ning–Liu)](decorrelated-score.md)

## Online / streaming

- [Stochastic Gradient Descent (SGD)](sgd.md)
- [Implicit SGD (ISGD)](implicit-sgd.md)
- [AdaGrad](adagrad.md)
- [FOBOS](fobos.md)
- [Regularized Dual Averaging (RDA)](rda.md)
- [Truncated Gradient](truncated-gradient.md)
- [Renewable Estimation (streaming GLM)](renewable-glm.md)

---

*The lists above are the planned initial set; cards are added incrementally. Missing links
point to cards not yet written.*
