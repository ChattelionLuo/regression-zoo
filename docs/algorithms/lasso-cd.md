---
id: lasso-cd
name: "Lasso via Coordinate Descent"
aliases: ["glmnet lasso", "pathwise coordinate descent", "shooting algorithm"]
family: penalized-batch
regime: [high-dim, low-dim]
penalty: lasso
link_support: [identity, logit, log]
output: path
year: 2007
refs: [tibshirani1996regression, fhht2007, zhao2006model]
status: reviewed
---

# Lasso via Coordinate Descent

!!! info "At a glance"
    **Family:** penalized-batch · **Regime:** high-dim / low-dim · **Penalty:** lasso ·
    **Output:** path over $\lambda$ · **Links:** identity, logit, log · **Status:** reviewed ·
    **Refs:** [`Tibshirani, 1996`](#ref-tibshirani1996regression) · [`Friedman et al., 2007`](#ref-fhht2007)

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family);
  Gaussian/identity is the canonical case, logistic/Poisson handled via the IRLS outer loop below.
- High- or low-dimensional; sparsity $\lVert\beta^\star\rVert_0=s\ll p$ is the motivating regime.
- Columns of $X$ standardized to mean $0$, unit variance; $y$ centered (Gaussian). Intercept unpenalized.

## Estimator / objective

The $\ell_1$-penalized template:

$$
\widehat\beta(\lambda) \;=\; \arg\min_{\beta\in\mathbb{R}^p}\;
\mathcal L(\beta) \;+\; \lambda\lVert\beta\rVert_1 ,
\qquad
\mathcal L(\beta)=\frac{1}{2n}\lVert y-X\beta\rVert_2^2 \;\text{(Gaussian)} .
$$

For a general GLM, $\mathcal L$ is the mean negative log-likelihood (logistic, Poisson, …).

## Algorithm

**Gaussian — cyclic coordinate descent.** With standardized columns ($\tfrac1n\lVert X_{\cdot j}\rVert_2^2=1$),
each coordinate has a closed-form soft-threshold update. Let the partial residual be
$r^{(j)} = y - \sum_{k\ne j} X_{\cdot k}\beta_k$. Then

$$
\beta_j \;\leftarrow\; \mathcal S_{\lambda}\!\Big(\tfrac1n X_{\cdot j}^\top r^{(j)}\Big),
\qquad
\mathcal S_{\lambda}(z)=\operatorname{sign}(z)\,(|z|-\lambda)_+ .
$$

```text
Input: X (standardized), y (centered), λ-grid λ_max>...>λ_min
Warm starts along the grid (pathwise):
for λ in grid:                       # decreasing
    repeat until convergence:
        for j = 1..p:
            r = y - X β + X[:,j] β_j          # partial residual
            β_j = Soft( (1/n) X[:,j]ᵀ r , λ )  # soft-threshold
    record β(λ)
Return path {β(λ)}
```

- $\lambda_{\max}=\tfrac1n\lVert X^\top y\rVert_\infty$ (smallest $\lambda$ with $\widehat\beta=0$);
  grid is log-spaced down to $\lambda_{\min}=\epsilon\,\lambda_{\max}$.
- **Active-set / strong rules** restrict cycling to likely-nonzero coordinates for speed.

**General GLM — penalized IRLS (outer) + coordinate descent (inner).** Repeat: form the
quadratic approximation of $\mathcal L$ at the current $\beta$ (working response
$z_i=\eta_i+(y_i-\mu_i)g'(\mu_i)$, weights $w_i$), then run weighted coordinate descent on the
penalized weighted least squares problem.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ | path | selected by CV (`lambda.min` / `lambda.1se`), AIC/BIC, or fixed |
| grid length | 100 | log-spaced $\lambda_{\max}\to\epsilon\lambda_{\max}$, $\epsilon=10^{-3}$ (or $10^{-2}$ if $p>n$) |
| standardize | true | columns to unit variance; coefficients returned on original scale |
| intercept | true, unpenalized | |
| tol | $10^{-7}$ | convergence on max coordinate change |
| family/link | gaussian/identity | also binomial/logit, poisson/log via IRLS |

## Mapping to framework

- **Input:** $X, y$, link; regularization $\lambda$ (or request full path).
- **Output:** $\widehat\beta(\lambda)$ — a single point or the whole path.
- **Links:** identity (LS inner loop), logit, log (IRLS outer loop).
- **Preprocessing:** standardize $X$; center $y$ (Gaussian) or fit unpenalized intercept (GLM).

## Complexity

- Per full cycle: $O(np)$ (Gaussian, dense), or $O(n\,\lvert\text{active set}\rvert)$ with active-set tricks.
- Whole path of $L$ values with warm starts: typically near $O(npL)$ in practice.
- Memory $O(np)$ (or $O(\text{nnz})$ for sparse $X$).

## Statistical guarantees

- **Estimation:** under a restricted-eigenvalue / compatibility condition and
  $\lambda\asymp\sigma\sqrt{\log p / n}$, $\lVert\widehat\beta-\beta^\star\rVert_1=O_P(s\sqrt{\log p/n})$.
- **Selection:** support recovery under the irrepresentable/beta-min conditions
  [`Zhao and Yu, 2006`](#ref-zhao2006model).
- Coordinate descent converges to a global optimum (convex objective, separable penalty).

## Variants & related

- [Elastic Net](elastic-net.md) · [Adaptive Lasso](adaptive-lasso.md) ·
  [Group Lasso](group-lasso.md) · [Fused Lasso](fused-lasso.md)
- [LARS](lars.md) — exact path; [FISTA](fista.md)/[ISTA](ista.md) — proximal-gradient solvers.
- [Debiased Lasso](debiased-lasso.md) — inference built on this estimator.

## References

- <a id="ref-tibshirani1996regression"></a> <a href="https://doi.org/10.1111/j.2517-6161.1996.tb02080.x" class="ref-link" target="_blank" rel="noopener noreferrer">Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *J. R. Stat. Soc. Ser. B*, 58(1):267--288.</a>
- <a id="ref-fhht2007"></a> <a href="https://doi.org/10.1214/07-AOAS131" class="ref-link" target="_blank" rel="noopener noreferrer">Friedman, J., Hastie, T., Höfling, H., and Tibshirani, R. (2007). Pathwise coordinate optimization. *Ann. Appl. Stat.*, 1(2):302--332.</a>
- <a id="ref-zhao2006model"></a> <a href="https://jmlr.org/papers/v7/zhao06a.html" class="ref-link" target="_blank" rel="noopener noreferrer">Zhao, P. and Yu, B. (2006). On model selection consistency of Lasso. *J. Mach. Learn. Res.*, 7:2541--2563.</a>
