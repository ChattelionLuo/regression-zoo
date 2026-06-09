---
hide:
  - toc
---

<div class="arena-hero" markdown>

<span class="arena-eyebrow">Linear Model Arena · Phase II</span>

# The Arena — Solvers in the Wild

<p class="arena-lede" markdown>
Every solver, faithfully implemented. Every dataset, rigorously tested. For the first time, you can watch how classical, penalized, and online methods converge — or diverge — from the same starting point.
</p>

<div class="arena-cta" markdown>
[Embedding maps](embeddings.md){ .arena-btn .arena-btn--primary }
[Coefficient heatmaps](heatmaps.md){ .arena-btn .arena-btn--ghost }
</div>

</div>

## What the Arena does

Phase II takes the 20 mathematically-documented solvers from Phase I and gives each one a faithful Python implementation.
All solvers share the same interface:

```python
result = SolverCls(config={...}).fit(X, y, link="identity")  # → FitResult
beta_hat = result.beta_hat  # shape (p,)
```

They are then run across **16 real and large-scale synthetic datasets** — regression and logistic — sweeping **50+ hyperparameter configurations per solver** where applicable (lambda grids, step sizes, decay schedules, group structures). Each *(solver, config)* pair produces one coefficient vector $\hat\beta$.

The output is two classes of visualization:

<div class="grid cards" markdown>

-   :material-chart-scatter-plot:{ .lg } &nbsp; **Nonlinear embedding maps**

    ---

    Unit-normalised coefficient vectors $\hat\beta / \|\hat\beta\|$ projected into 2D via **t-SNE** and **UMAP**. Solvers that produce algorithmically similar solutions cluster together. Hover to see the solver name and exact hyperparameter configuration.

    [:octicons-arrow-right-24: View embedding maps](embeddings.md)

-   :material-grid:{ .lg } &nbsp; **Coefficient heatmaps**

    ---

    All runs for one dataset laid out as a matrix — rows are runs sorted by solver family, columns are features. The colour shows $\hat\beta_j$. Which features does each family activate? How does regularization strength change the pattern?

    [:octicons-arrow-right-24: View heatmaps](heatmaps.md)

</div>

---

## Solver implementations

All 20 solvers are implemented under `src/glmzoo/solvers/`, grouped by family. Each implementation is a faithful translation of the source algorithm — notation, hyperparameter names, and update rules match the corresponding documentation card.

<div class="algo-grid" markdown>

[OLS](../algorithms/ols.md){ .algo-item }
[GLM-IRLS](../algorithms/glm-irls.md){ .algo-item }
[LARS](../algorithms/lars.md){ .algo-item }
[Ridge](../algorithms/ridge.md){ .algo-item }
[Lasso-CD](../algorithms/lasso-cd.md){ .algo-item }
[Elastic Net](../algorithms/elastic-net.md){ .algo-item }
[Adaptive Lasso](../algorithms/adaptive-lasso.md){ .algo-item }
[SCAD](../algorithms/scad-lla.md){ .algo-item }
[MCP-CD](../algorithms/mcp-cd.md){ .algo-item }
[Group Lasso](../algorithms/group-lasso.md){ .algo-item }
[Fused Lasso](../algorithms/fused-lasso.md){ .algo-item }
[ISTA](../algorithms/ista.md){ .algo-item }
[FISTA](../algorithms/fista.md){ .algo-item }
[SGD](../algorithms/sgd.md){ .algo-item }
[Implicit SGD](../algorithms/implicit-sgd.md){ .algo-item }
[AdaGrad](../algorithms/adagrad.md){ .algo-item }
[FOBOS](../algorithms/fobos.md){ .algo-item }
[RDA](../algorithms/rda.md){ .algo-item }
[Truncated Gradient](../algorithms/truncated-gradient.md){ .algo-item }
[Renewable GLM](../algorithms/renewable-glm.md){ .algo-item }

</div>

---

## Datasets

16 datasets spanning real-world and synthetic regression and logistic tasks:

| Dataset | Source | n | p | Task |
|---|---|---|---|---|
| Diabetes | sklearn / UCI | 442 | 10 | Regression |
| Breast Cancer | sklearn / UCI | 569 | 30 | Logistic |
| Digits (≥5) | sklearn / NIST | 1 797 | 64 | Logistic |
| Fair Affairs | statsmodels | 2 000 | 8 | Regression |
| RAND HIE | statsmodels | 2 000 | 9 | Regression |
| STAR98 pass | statsmodels | 303 | 21 | Logistic |
| ANES96 vote | statsmodels | 944 | 10 | Logistic |
| Mode Choice | statsmodels | 840 | 8 | Logistic |
| Synth Reg Sparse200 | synthetic | 2 000 | 200 | Regression |
| Synth Reg Dense100 | synthetic | 2 000 | 100 | Regression |
| Synth Reg Corr150 | synthetic | 2 000 | 150 | Regression |
| Synth Reg HighDim300 | synthetic | 2 000 | 300 | Regression |
| Synth Friedman1 | synthetic | 2 000 | 10 | Regression |
| Synth Logit Dense100 | synthetic | 2 000 | 100 | Logistic |
| Synth Logit Sparse200 | synthetic | 2 000 | 200 | Logistic |
| Synth Logit Noisy150 | synthetic | 2 000 | 150 | Logistic |

Real datasets (sklearn, statsmodels) are standardised and subsampled to n ≤ 2 000. Synthetic datasets are generated with controlled sparsity, correlation structure, and noise.
