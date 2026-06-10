---
id: adaptive-lasso
name: "Adaptive Lasso"
aliases: ["adalasso", "weighted lasso", "reweighted L1"]
family: penalized-batch
regime: [low-dim, high-dim]
penalty: adaptive-lasso
link_support: [identity, logit, log]
output: path
year: 2006
refs: [buhlmann2011statistics, tibshirani1996regression, fhht2007]
status: draft
---

# Adaptive Lasso

!!! info "At a glance"
    **Family:** penalized-batch · **Regime:** low-dim / high-dim · **Penalty:** adaptive-lasso ·
    **Output:** path over $\lambda$ · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** [`Buhlmann and van de Geer, 2011`](#ref-buhlmann2011statistics) · [`Tibshirani, 1996`](#ref-tibshirani1996regression)

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family);
  Gaussian/identity is canonical, logistic/Poisson via the IRLS outer loop.
- Originally proposed for the low-dimensional regime where a $\sqrt n$-consistent initial
  estimator (e.g. OLS) exists; extends to high dimensions when $\widehat\beta_{\text{init}}$ is a
  ridge or marginal-regression estimate.
- Columns of $X$ standardized to unit variance; $y$ centered (Gaussian). Intercept unpenalized.
- Sparsity $\lVert\beta^\star\rVert_0=s$ assumed; relies on an initial estimate whose nonzero
  coefficients are bounded away from $0$ (beta-min).

## Estimator / objective

The adaptive lasso replaces the uniform $\ell_1$ penalty by a **weighted** $\ell_1$ penalty whose
weights are computed from a preliminary estimate $\widehat\beta_{\text{init}}$:

$$
\widehat\beta(\lambda) \;=\; \arg\min_{\beta\in\mathbb{R}^p}\;
\frac{1}{2n}\lVert y-X\beta\rVert_2^2
\;+\; \lambda \sum_{j=1}^p w_j\,|\beta_j|,
\qquad
w_j = \frac{1}{|\widehat\beta_{\text{init},j}|^{\gamma}},\;\; \gamma>0 .
$$

For a general GLM, replace the Gaussian loss by the mean negative log-likelihood $\mathcal L(\beta)$.
Large initial coefficients get **small** weights (penalized lightly, kept), while near-zero ones
get **large** weights (penalized heavily, driven to exactly $0$). This data-driven asymmetry is
what buys the oracle property that the plain lasso lacks.

## Algorithm

The key observation is that the weighted problem **reduces to an ordinary lasso** by rescaling
the columns. Define rescaled predictors $\tilde X_{\cdot j} = X_{\cdot j}/w_j$ and coefficients
$\tilde\beta_j = w_j\beta_j$. Then

$$
\frac{1}{2n}\lVert y-X\beta\rVert_2^2 + \lambda\sum_j w_j|\beta_j|
= \frac{1}{2n}\lVert y-\tilde X\tilde\beta\rVert_2^2 + \lambda\lVert\tilde\beta\rVert_1 ,
$$

an ordinary lasso in $(\tilde X,\tilde\beta)$. Solve it by [coordinate descent](lasso-cd.md),
then map back $\widehat\beta_j = \widehat{\tilde\beta}_j / w_j$.

```text
Input: X (standardized), y (centered), λ-grid, exponent γ, initial estimator type
1. Compute β_init  (OLS if n>p; ridge or marginal regression if p≥n)
2. Weights:  w_j = 1 / |β_init,j|^γ        # w_j = ∞ where β_init,j = 0  → β_j forced to 0
3. Rescale:  X̃[:,j] = X[:,j] / w_j
4. Solve ordinary lasso on (X̃, y) by coordinate descent over the λ-grid:
       for λ in grid (decreasing), warm-started:
           repeat until convergence:
               for j with w_j < ∞:
                   r = y - X̃ β̃ + X̃[:,j] β̃_j
                   β̃_j = Soft( (1/n) X̃[:,j]ᵀ r , λ )
5. Map back:  β_j(λ) = β̃_j(λ) / w_j
Return path {β(λ)}
```

Coordinates with $\widehat\beta_{\text{init},j}=0$ have $w_j=\infty$ and are dropped a priori
($\beta_j\equiv 0$), giving an automatic screening step. For GLMs the inner solve is the
penalized IRLS + coordinate-descent loop applied to $\tilde X$.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ | path | selected by CV (`lambda.min` / `lambda.1se`), AIC/BIC, or fixed |
| $\gamma$ (weight exponent) | 1 | $\gamma\in\{0.5,1,2\}$ typical; larger $\gamma$ ⇒ stronger asymmetry; tuned with $\lambda$ by CV |
| $\widehat\beta_{\text{init}}$ | OLS ($n>p$) | ridge or univariate/marginal regression when $p\ge n$ |
| standardize | true | columns to unit variance |
| intercept | true, unpenalized | |
| tol | $10^{-7}$ | convergence on max coordinate change |
| family/link | gaussian/identity | also binomial/logit, poisson/log via IRLS |

The pair $(\gamma,\lambda)$ (and the choice of initial estimator) are usually selected jointly by
cross-validation.

## Mapping to framework

- **Input:** $X, y$, link; weight exponent $\gamma$, initial estimator, regularization $\lambda$.
- **Output:** $\widehat\beta(\lambda)$ — a point or the whole path, on the original scale.
- **Links:** identity (LS inner loop), logit, log (IRLS outer loop).
- **Preprocessing:** standardize $X$; center $y$ (Gaussian). Requires computing
  $\widehat\beta_{\text{init}}$ before the main solve.

## Complexity

- Initial estimator: $O(np^2+p^3)$ for OLS/ridge, or $O(np)$ for marginal regression.
- Main solve: identical to lasso coordinate descent — per cycle $O(np)$, full path with warm
  starts near $O(npL)$; the $w_j=\infty$ screening shrinks the working set.
- Memory $O(np)$ (or $O(\text{nnz})$ for sparse $X$).

## Statistical guarantees

- **Oracle property** (Zou, 2006). With a $\sqrt n$-consistent $\widehat\beta_{\text{init}}$ and
  $\lambda_n$ chosen so that $\lambda_n/\sqrt n\to 0$ and $\lambda_n n^{(\gamma-1)/2}\to\infty$,
  the adaptive lasso (i) selects the true support with probability $\to 1$ and (ii) estimates the
  nonzero coefficients with the same asymptotic distribution as if the true support were known.
  The ordinary lasso does **not** enjoy this property in general.
- High-dimensional consistency holds under restricted-eigenvalue/compatibility conditions and a
  suitable initial estimator; see [`Bühlmann and van de Geer, 2011`](#ref-buhlmann2011statistics).

## Variants & related

- [Lasso](lasso-cd.md) — the uniform-weight special case ($w_j\equiv 1$).
- [Elastic Net](elastic-net.md) — **adaptive elastic net** combines weights with an $\ell_2$ term.
- [SCAD](scad-lla.md) / [MCP](mcp-cd.md) — nonconvex penalties achieving the oracle property directly.
- Iterating the reweighting step yields connections to nonconvex / reweighted-$\ell_1$ schemes.

## References

- <a id="ref-zou2006adaptive"></a> <a href="https://doi.org/10.1198/016214506000000735" class="ref-link" target="_blank" rel="noopener noreferrer">Zou, H. (2006). The adaptive lasso and its oracle properties. *J. Amer. Statist. Assoc.*, 101(476):1418--1429.</a>
- <a id="ref-buhlmann2011statistics"></a> <a href="https://link.springer.com/book/10.1007/978-3-642-20192-9" class="ref-link" target="_blank" rel="noopener noreferrer">Bühlmann, P. and van de Geer, S. (2011). *Statistics for High-Dimensional Data: Methods, Theory and Applications*. Springer.</a>
- <a id="ref-tibshirani1996regression"></a> <a href="https://doi.org/10.1111/j.2517-6161.1996.tb02080.x" class="ref-link" target="_blank" rel="noopener noreferrer">Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *J. R. Stat. Soc. Ser. B*, 58(1):267--288.</a>
- <a id="ref-fhht2007"></a> <a href="https://doi.org/10.1214/07-AOAS131" class="ref-link" target="_blank" rel="noopener noreferrer">Friedman, J., Hastie, T., Höfling, H., and Tibshirani, R. (2007). Pathwise coordinate optimization. *Ann. Appl. Stat.*, 1(2):302--332.</a>
