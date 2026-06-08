---
id: elastic-net
name: "Elastic Net"
aliases: ["enet", "glmnet elastic net", "L1+L2 regularization"]
family: penalized-batch
regime: [high-dim, low-dim]
penalty: elastic-net
link_support: [identity, logit, log]
output: path
year: 2005
refs: [zou2005regularization, fhht2007, tibshirani1996regression]
status: draft
---

# Elastic Net

!!! info "At a glance"
    **Family:** penalized-batch · **Regime:** high-dim / low-dim · **Penalty:** elastic-net ·
    **Output:** path over $\lambda$ · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** [`Zou and Hastie, 2005`](#ref-zou2005regularization) · [`Friedman et al., 2007`](#ref-fhht2007)

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family);
  Gaussian/identity is canonical, logistic/Poisson handled via the IRLS outer loop below.
- High- or low-dimensional. The method targets the **$p \gg n$ with correlated predictors**
  regime where pure lasso is unstable: lasso saturates at $n$ selected variables and arbitrarily
  picks one from a correlated group.
- Columns of $X$ standardized to mean $0$, unit variance; $y$ centered (Gaussian). Intercept unpenalized.
- Sparsity $\lVert\beta^\star\rVert_0=s$ assumed in the high-dimensional regime, but exact
  sparsity is not required — the $\ell_2$ part keeps grouped variables in together.

## Estimator / objective

The elastic net mixes the lasso ($\ell_1$) and ridge ($\ell_2$) penalties through a mixing
parameter $\alpha\in[0,1]$:

$$
\hat\beta(\lambda,\alpha) \;=\; \arg\min_{\beta\in\mathbb{R}^p}\;
\frac{1}{2n}\lVert y-X\beta\rVert_2^2
\;+\; \lambda\Big( \alpha\lVert\beta\rVert_1 + \frac{1-\alpha}{2}\lVert\beta\rVert_2^2 \Big).
$$

For a general GLM, replace the Gaussian loss by the mean negative log-likelihood $\mathcal L(\beta)$.
The limiting cases are $\alpha=1$ (pure [lasso](lasso-cd.md)) and $\alpha=0$ (pure
[ridge](ridge.md)). The combined penalty is **strictly convex** for $\alpha<1$, so the solution
is unique even when columns of $X$ are collinear.

## Algorithm

**Gaussian — cyclic coordinate descent.** With standardized columns
($\tfrac1n\lVert X_{\cdot j}\rVert_2^2=1$), each coordinate has a closed form combining
soft-thresholding (from the $\ell_1$ term) with a ridge shrinkage denominator (from the $\ell_2$
term). Let the partial residual be $r^{(j)} = y - \sum_{k\ne j} X_{\cdot k}\beta_k$. Then

$$
\beta_j \;\leftarrow\;
\frac{\mathcal S_{\lambda\alpha}\!\big(\tfrac1n X_{\cdot j}^\top r^{(j)}\big)}
     {1 + \lambda(1-\alpha)},
\qquad
\mathcal S_{t}(z)=\operatorname{sign}(z)\,(|z|-t)_+ .
$$

The numerator soft-thresholds at level $\lambda\alpha$; the denominator $1+\lambda(1-\alpha)$
is the proximal shrinkage induced by the ridge term.

```text
Input: X (standardized), y (centered), λ-grid λ_max>...>λ_min, mixing α
Warm starts along the grid (pathwise):
for λ in grid:                          # decreasing
    repeat until convergence:
        for j = 1..p:
            r = y - X β + X[:,j] β_j               # partial residual r^(j)
            z = (1/n) X[:,j]ᵀ r
            β_j = Soft(z, λ·α) / (1 + λ·(1-α))     # soft-threshold then ridge shrink
    record β(λ)
Return path {β(λ)}
```

- $\lambda_{\max}=\tfrac1{n\alpha}\lVert X^\top y\rVert_\infty$ (smallest $\lambda$ giving
  $\hat\beta=0$ for given $\alpha>0$); grid log-spaced down to $\lambda_{\min}=\epsilon\,\lambda_{\max}$.
- **Active-set / strong rules** restrict cycling to likely-nonzero coordinates.

**General GLM — penalized IRLS (outer) + coordinate descent (inner).** Form the quadratic
approximation of $\mathcal L$ at the current $\beta$ (working response
$z_i=\eta_i+(y_i-\mu_i)g'(\mu_i)$, weights $w_i$), then run the weighted version of the update
above on the penalized weighted least squares problem.

**Naive vs corrected (rescaled) elastic net.** The raw coordinate solution is the *naive*
elastic net, which applies a double amount of shrinkage ($\ell_1$ then $\ell_2$) and can
over-shrink. Zou & Hastie (2005) propose the *corrected* elastic net, rescaling

$$
\hat\beta^{\text{enet}} = (1+\lambda(1-\alpha))\,\hat\beta^{\text{naive}},
$$

which undoes the ridge contraction while keeping the grouping/variable-selection behaviour.
Modern `glmnet`-style parameterizations fold this scaling into the penalty definition.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ | path | selected by CV (`lambda.min` / `lambda.1se`), AIC/BIC, or fixed |
| $\alpha$ (mixing) | 0.5 | $1$ = lasso, $0$ = ridge; often itself tuned on a small grid by CV |
| grid length | 100 | log-spaced $\lambda_{\max}\to\epsilon\lambda_{\max}$, $\epsilon=10^{-3}$ ($10^{-2}$ if $p>n$) |
| standardize | true | columns to unit variance; coefficients returned on original scale |
| intercept | true, unpenalized | |
| correction | true | rescale naive → corrected estimate |
| tol | $10^{-7}$ | convergence on max coordinate change |
| family/link | gaussian/identity | also binomial/logit, poisson/log via IRLS |

Selecting $(\lambda,\alpha)$ is typically a 2-D cross-validation over a small $\alpha$ grid
$\{0,0.1,\dots,1\}$ with a full $\lambda$ path for each.

## Mapping to framework

- **Input:** $X, y$, link; regularization $\lambda$ and mixing $\alpha$ (or request the full path).
- **Output:** $\hat\beta(\lambda,\alpha)$ — a single point or the whole path over $\lambda$.
- **Links:** identity (LS inner loop), logit, log (IRLS outer loop).
- **Preprocessing:** standardize $X$; center $y$ (Gaussian) or fit an unpenalized intercept (GLM).

## Complexity

- Per full cycle: $O(np)$ (Gaussian, dense), or $O(n\,\lvert\text{active set}\rvert)$ with active-set tricks.
- Whole path of $L$ values with warm starts: typically near $O(npL)$ in practice; multiply by the
  number of $\alpha$ values when tuning the mixing parameter.
- Memory $O(np)$ (or $O(\text{nnz})$ for sparse $X$).

## Statistical guarantees

- **Grouping effect.** For two predictors with sample correlation $\rho$, Zou & Hastie (2005)
  bound the coefficient difference: $\;|\hat\beta_j-\hat\beta_k| \le C\sqrt{2(1-\rho)}\,$,
  so highly correlated predictors receive nearly equal coefficients (and enter/leave the model
  together) — unlike the lasso, which selects one arbitrarily.
- Strict convexity for $\alpha<1$ gives a **unique** solution even under collinearity; the model
  can select more than $n$ variables (lasso cannot).
- Estimation/selection consistency follows lasso-type analyses under restricted-eigenvalue /
  compatibility conditions with $\lambda\asymp\sigma\sqrt{\log p/n}$.

## Variants & related

- [Lasso](lasso-cd.md) ($\alpha=1$) · [Ridge](ridge.md) ($\alpha=0$) — the two endpoints.
- [Adaptive Lasso](adaptive-lasso.md) · [Group Lasso](group-lasso.md) ·
  [Fused Lasso](fused-lasso.md) — other structured penalties.
- **Adaptive elastic net** — combine data-driven weights with the $\ell_1+\ell_2$ mix.

## References

- <a id="ref-zou2005regularization"></a> Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. *J. R. Stat. Soc. Ser. B*, 67(2):301--320.
- <a id="ref-fhht2007"></a> Friedman, J., Hastie, T., Höfling, H., and Tibshirani, R. (2007). Pathwise coordinate optimization. *Ann. Appl. Stat.*, 1(2):302--332.
- <a id="ref-tibshirani1996regression"></a> Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *J. R. Stat. Soc. Ser. B*, 58(1):267--288.
