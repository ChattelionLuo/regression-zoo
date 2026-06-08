<p class="arena-kicker">Phase I · Documentation</p>

# Algorithm Catalogue

The library of algorithm cards, grouped by [family](../framework/taxonomy.md). Each entry maps
$(X, y, g) \mapsto \hat\beta$ under the [unified notation](../framework/notation.md).

!!! note "Status legend"
    **reviewed** — math checked against source · **draft** — written, pending review ·
    **stub** — scaffold only.

## Classical (low-dimensional regression & GLMs)

<div class="algo-grid">
<a class="algo-item" href="ols/">Ordinary Least Squares (OLS)</a>
<a class="algo-item" href="ridge/">Ridge Regression</a>
<a class="algo-item" href="glm-irls/">GLM via IRLS / Fisher Scoring</a>
</div>

## Penalized (batch)

<div class="algo-grid">
<a class="algo-item" href="lasso-cd/">Lasso via Coordinate Descent</a>
<a class="algo-item" href="elastic-net/">Elastic Net</a>
<a class="algo-item" href="adaptive-lasso/">Adaptive Lasso</a>
<a class="algo-item" href="scad-lla/">SCAD via Local Linear Approximation</a>
<a class="algo-item" href="mcp-cd/">MCP via Coordinate Descent</a>
<a class="algo-item" href="group-lasso/">Group Lasso</a>
<a class="algo-item" href="fused-lasso/">Fused Lasso</a>
</div>

## Path / homotopy

<div class="algo-grid">
<a class="algo-item" href="lars/">Least Angle Regression (LARS)</a>
</div>

## First-order / proximal solvers

<div class="algo-grid">
<a class="algo-item" href="ista/">ISTA</a>
<a class="algo-item" href="fista/">FISTA</a>
</div>

## High-dimensional inference

<div class="algo-grid">
<a class="algo-item" href="debiased-lasso/">Debiased / Desparsified Lasso</a>
<a class="algo-item" href="decorrelated-score/">Decorrelated Score (Ning–Liu)</a>
</div>

## Online / streaming

<div class="algo-grid">
<a class="algo-item" href="sgd/">Stochastic Gradient Descent (SGD)</a>
<a class="algo-item" href="implicit-sgd/">Implicit SGD (ISGD)</a>
<a class="algo-item" href="adagrad/">AdaGrad</a>
<a class="algo-item" href="fobos/">FOBOS</a>
<a class="algo-item" href="rda/">Regularized Dual Averaging (RDA)</a>
<a class="algo-item" href="truncated-gradient/">Truncated Gradient</a>
<a class="algo-item" href="renewable-glm/">Renewable Estimation (streaming GLM)</a>
</div>

---

*The lists above are the planned initial set; cards are added incrementally.*
