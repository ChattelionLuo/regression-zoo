---
id: ridge
name: "Ridge Regression"
aliases: ["Tikhonov regularization", "L2-penalized least squares", "weight decay"]
family: classical
regime: [low-dim, high-dim]
penalty: ridge
link_support: [identity, logit, log]
output: point
year: 1970
refs: [Nocedal2006, zou2005regularization, buhlmann2011statistics]
status: draft
---

# Ridge Regression

!!! info "At a glance"
    **Family:** classical · **Regime:** low-dim / high-dim · **Penalty:** ridge ($\ell_2$) ·
    **Output:** point · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** [`Nocedal and Wright, 2006`](#ref-Nocedal2006) · [`Zou and Hastie, 2005`](#ref-zou2005regularization) · [`Buhlmann and van de Geer, 2011`](#ref-buhlmann2011statistics)

## Setting & assumptions

- Gaussian family with **identity link** in the canonical case: $\;y = X\beta^\star + \varepsilon$,
  $\mathbb{E}[\varepsilon\mid X]=0$; the penalized-likelihood form extends to any GLM.
- Works in **low-dimensional** ($n\ge p$) and **high-dimensional** ($p\gtrsim n$) regimes:
  the penalty makes $X^\top X + 2n\lambda I \succ 0$ even when $X^\top X$ is singular or
  ill-conditioned, so a unique solution always exists.
- Columns of $X$ are **standardized** (centered, unit variance / unit $\ell_2$-norm) so that the
  single scalar $\lambda$ shrinks comparable coordinates equally; the intercept is left unpenalized.
- No sparsity is assumed or produced — ridge shrinks but does not zero out coefficients.

## Estimator / objective

Ridge is the $\ell_2$-penalized template ($P(\beta)=\lVert\beta\rVert_2^2$) with the Gaussian loss:

$$
\widehat\beta(\lambda) \;=\; \arg\min_{\beta\in\mathbb{R}^p}\;
\frac{1}{2n}\,\lVert y - X\beta\rVert_2^2 \;+\; \lambda\,\lVert\beta\rVert_2^2 .
$$

Setting the gradient $-\tfrac1n X^\top(y-X\beta) + 2\lambda\beta$ to zero gives the
**regularized normal equations** $\big(X^\top X + 2n\lambda I\big)\beta = X^\top y$, with the
unique closed form

$$
\boxed{\;\widehat\beta(\lambda) = \big(X^\top X + 2n\lambda I\big)^{-1} X^\top y\;}
$$

*(Arena note — scaling convention.)* Because we use the $\tfrac{1}{2n}$-normalized loss together
with the penalty $\lambda\lVert\beta\rVert_2^2$, the ridge term entering the normal equations is
$2n\lambda I$. Authors who write the loss as $\tfrac12\lVert y-X\beta\rVert_2^2$ with penalty
$\tfrac\lambda2\lVert\beta\rVert_2^2$ obtain the more familiar $(X^\top X+\lambda I)^{-1}X^\top y$;
the two agree under $\lambda_{\text{here}} = \lambda_{\text{there}}/(2n)$. State the convention
whenever reporting a numerical $\lambda$.

**GLM / penalized-likelihood generalization.** For a general GLM the estimator solves

$$
\widehat\beta(\lambda) \;=\; \arg\min_{\beta}\; \mathcal L(\beta) + \lambda\lVert\beta\rVert_2^2 ,
$$

with $\mathcal L$ the mean negative log-likelihood (logistic, Poisson, …); the penalized score
equation is $X^\top(y-\mu(\beta)) = 2n\lambda\,\beta$.

## Algorithm

**SVD solution and singular-value shrinkage.** Let $X = U\Sigma V^\top$ be the (thin) SVD with
singular values $\sigma_1\ge\cdots\ge\sigma_r>0$. Then

$$
\widehat\beta(\lambda) = \sum_{k} v_k\,\frac{\sigma_k}{\sigma_k^2 + 2n\lambda}\, u_k^\top y ,
$$

so each component of the least-squares solution is multiplied by the **shrinkage factor**
$\sigma_k^2/(\sigma_k^2 + 2n\lambda)\in(0,1)$: directions with small $\sigma_k$ (the unstable ones)
are damped most. This is exactly **Tikhonov regularization** of the ill-posed inverse problem.

```text
Input: X (n×p, standardized), y (n, centered), λ ≥ 0
Closed form (Cholesky, low-dim p < n):
  1. A = XᵀX + 2nλ I            # p×p, SPD for any λ > 0
  2. b = Xᵀ y
  3. solve A β = b  via Cholesky factorization
Closed form (SVD, robust / high-dim):
  1. X = U Σ Vᵀ
  2. β = Σ_k v_k · σ_k/(σ_k² + 2nλ) · (u_kᵀ y)
Return β̂(λ)
```

**Penalized IRLS for GLMs.** For non-identity links, ridge is solved by the same IRLS loop as a
GLM but with a ridge term added to each weighted least-squares step:

```text
repeat until convergence:
    η = Xβ;  μ = g⁻¹(η)
    z_i = η_i + (y_i - μ_i) g'(μ_i)                 # working response
    W   = diag( 1 / (g'(μ_i)² V(μ_i)) )             # IRLS weights
    β  <- (Xᵀ W X + 2nλ I)⁻¹ Xᵀ W z                 # ridge-penalized WLS update
```

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ | by CV | non-negative ridge level; selected by $k$-fold CV or **GCV** (below) |
| standardize | true | center + scale columns; coefficients returned on the original scale |
| intercept | true, unpenalized | fit by centering $y$ (Gaussian) or as a free column (GLM) |
| solver | Cholesky / SVD | SVD gives the whole $\lambda$-path cheaply once $X=U\Sigma V^\top$ is formed |
| tol / max-iter | $10^{-7}$ / 100 | IRLS stopping for GLM links |

**$\lambda$ selection.** Generalized cross-validation minimizes
$\mathrm{GCV}(\lambda)=\tfrac1n\lVert y - X\widehat\beta(\lambda)\rVert_2^2 / \big(1-\mathrm{df}(\lambda)/n\big)^2$,
a rotation-invariant approximation to leave-one-out CV that reuses the SVD. Ordinary $k$-fold CV
on prediction error is the common alternative.

**Effective degrees of freedom.** With hat matrix $H_\lambda = X(X^\top X + 2n\lambda I)^{-1}X^\top$,

$$
\mathrm{df}(\lambda) = \operatorname{tr}(H_\lambda) = \sum_k \frac{\sigma_k^2}{\sigma_k^2 + 2n\lambda},
$$

decreasing from $p$ (at $\lambda=0$) toward $0$ as $\lambda\to\infty$; it quantifies model
complexity and feeds GCV.

## Mapping to framework

- **Input:** $X\in\mathbb{R}^{n\times p}$, $y\in\mathbb{R}^n$, link (`identity`/`logit`/`log`),
  ridge level $\lambda$.
- **Output:** point estimate $\widehat\beta(\lambda)=(X^\top X + 2n\lambda I)^{-1}X^\top y$ (identity),
  or the penalized-IRLS solution for other links.
- **Links:** identity (closed form); logit, log via penalized IRLS.
- **Preprocessing:** standardize $X$, center $y$ / keep intercept unpenalized; report the $\lambda$
  convention used.

## Complexity

- Cholesky route: $O(np^2)$ to form $X^\top X$ plus $O(p^3)$ to factor/solve; memory $O(p^2)$.
- SVD route: $O(np\min(n,p))$ once, after which every $\lambda$ on a grid costs only $O(pr)$
  — ideal for GCV/CV over a $\lambda$-path.
- High-dim ($p\gg n$): use the dual/woodbury form $\widehat\beta = X^\top(XX^\top + 2n\lambda I)^{-1}y$
  at $O(n^2 p + n^3)$.

## Statistical guarantees

- **Bias–variance trade-off.** $\widehat\beta(\lambda)$ is biased toward $0$ but has strictly smaller
  variance than OLS; for the Gaussian model there always exists a $\lambda>0$ whose mean-squared
  error beats OLS — the original motivation of Hoerl & Kennard (1970).
- **Existence/uniqueness.** The objective is strongly convex for any $\lambda>0$, so $\widehat\beta$ is
  unique even when $p>n$ or $X$ is rank-deficient.
- **Bayesian reading.** $\widehat\beta(\lambda)$ is the posterior mean under a Gaussian prior
  $\beta\sim\mathcal N(0,\tau^2 I)$ with $\tau^2 \propto 1/\lambda$.
- Ridge does **not** perform variable selection; for sparse recovery use $\ell_1$ penalties.

## Variants & related

- [OLS](ols.md) — the $\lambda\to 0$ limit (full-rank $X$).
- [Lasso via Coordinate Descent](lasso-cd.md) — $\ell_1$ analogue producing sparsity.
- [Elastic Net](elastic-net.md) — convex combination of ridge and lasso [`Zou and Hastie, 2005`](#ref-zou2005regularization).
- [GLM via IRLS](glm-irls.md) — the unpenalized engine reused inside ridge's penalized IRLS.

## References

- <a id="ref-hoerl1970ridge"></a> <a href="https://doi.org/10.1080/00401706.1970.10488634" class="ref-link" target="_blank" rel="noopener noreferrer">Hoerl, A. E. and Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, 12(1):55--67.</a>
- <a id="ref-Nocedal2006"></a> <a href="https://link.springer.com/book/10.1007/978-0-387-40065-5" class="ref-link" target="_blank" rel="noopener noreferrer">Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer.</a>
- <a id="ref-zou2005regularization"></a> <a href="https://doi.org/10.1111/j.1467-9868.2005.00503.x" class="ref-link" target="_blank" rel="noopener noreferrer">Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. *J. R. Stat. Soc. Ser. B*, 67(2):301--320.</a>
- <a id="ref-buhlmann2011statistics"></a> <a href="https://link.springer.com/book/10.1007/978-3-642-20192-9" class="ref-link" target="_blank" rel="noopener noreferrer">Bühlmann, P. and van de Geer, S. (2011). *Statistics for High-Dimensional Data: Methods, Theory and Applications*. Springer.</a>
