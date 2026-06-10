---
id: glm-irls
name: "GLM via IRLS / Fisher Scoring"
aliases: ["iteratively reweighted least squares", "Fisher scoring", "IWLS", "GLM MLE"]
family: classical
regime: [low-dim]
penalty: none
link_support: [any]
output: point
year: 1972
refs: [IWLS1987, Nocedal2006]
status: draft
---

# GLM via IRLS / Fisher Scoring

!!! info "At a glance"
    **Family:** classical · **Regime:** low-dim ($n>p$) · **Penalty:** none ·
    **Output:** point · **Links:** any differentiable $g$ · **Status:** draft ·
    **Refs:** [`Green, 1987`](#ref-IWLS1987) · [`Nocedal and Wright, 2006`](#ref-Nocedal2006)

## Setting & assumptions

- Responses follow a one-parameter
  [exponential dispersion family](../framework/notation.md#2-exponential-dispersion-family)
  with mean $\mu_i=g^{-1}(x_i^\top\beta)$ for a chosen monotone, differentiable **link** $g$
  (canonical or not), variance function $V(\mu)$, and dispersion $\phi$.
- Low-dimensional: $n>p$ and $X$ full column rank, so the Fisher information
  $X^\top W X$ is invertible and the MLE is well-defined.
- Observations conditionally independent given $X$; the linear predictor is correctly specified.
- No penalty ($P\equiv 0$): this is the plain maximum-likelihood estimator.

## Estimator / objective

The MLE maximizes the log-likelihood, equivalently **minimizes the mean negative log-likelihood**:

$$
\widehat\beta \;=\; \arg\min_{\beta\in\mathbb{R}^p}\;
\mathcal L(\beta) , \qquad
\mathcal L(\beta) = \frac1n\sum_{i=1}^n \ell_i(\beta),
$$

with $\ell_i$ the per-sample negative log-likelihood. The optimum solves the **score equation**

$$
S(\beta) \;=\; X^\top\!\big(y - \mu(\beta)\big) \;=\; 0 ,
$$

(for the canonical link; for a general link a diagonal derivative factor appears, which the IRLS
weights below absorb). There is no closed form in general, so $\widehat\beta$ is found iteratively.

## Algorithm

**Iteratively Reweighted Least Squares (IRLS) = Fisher scoring.** Each iteration forms a local
**working response** and **weights**, then solves a weighted least-squares problem:

$$
z_i = \eta_i + (y_i-\mu_i)\,g'(\mu_i), \qquad
W_i = \frac{1}{g'(\mu_i)^2\,V(\mu_i)}, \qquad
\beta \leftarrow (X^\top W X)^{-1} X^\top W z .
$$

```text
Input: X (n×p), y (n), link g, variance function V
Init:  μ_i = sensible start (e.g. y_i adjusted), η_i = g(μ_i),  β = 0
repeat (t = 1, 2, ...):
    η_i = x_iᵀ β
    μ_i = g⁻¹(η_i)
    z_i = η_i + (y_i - μ_i) · g'(μ_i)            # working / adjusted response
    W_i = 1 / ( g'(μ_i)² · V(μ_i) )              # IRLS weights (diagonal)
    β_new = (Xᵀ W X)⁻¹ Xᵀ W z                    # weighted least squares step
    if deviance increased:  β_new = β + ½(β_new - β)   # step-halving (repeat as needed)
    if ||β_new - β|| < tol:  break
    β = β_new
Return β̂   (and dispersion φ̂, deviance D)
```

**Newton–Raphson equivalence under the canonical link.** Newton's method updates
$\beta \leftarrow \beta - \big(\nabla^2\mathcal L\big)^{-1}\nabla\mathcal L$. Using the framework
identities $\nabla\mathcal L = -\tfrac1n X^\top(y-\mu)$ and $\nabla^2\mathcal L = \tfrac1n X^\top W X$,
the update becomes $\beta + (X^\top W X)^{-1}X^\top(y-\mu)$, which is **exactly** the IRLS step
$(X^\top W X)^{-1}X^\top W z$. For the canonical link the observed information **equals** the
expected (Fisher) information, so Newton–Raphson and Fisher scoring coincide; for non-canonical
links Fisher scoring replaces the observed Hessian by its expectation $X^\top W X$, which keeps the
weights positive and the step stable.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| link $g$ | canonical | any monotone differentiable link; non-canonical → Fisher scoring |
| init | $\mu_i\approx y_i$ | mean-start avoids boundary issues (e.g. $\mu_i=(y_i+\bar y)/2$ for binomial) |
| tol | $10^{-8}$ | convergence on $\lVert\Delta\beta\rVert$ or relative deviance change |
| max-iter | 25 | IRLS converges quadratically near the optimum |
| step-halving | on | backtrack $\beta+2^{-k}\Delta$ if deviance worsens / iterate diverges |
| dispersion $\phi$ | estimated | fixed at $1$ for binomial/Poisson; estimated for Gaussian/Gamma |

**Dispersion estimation.** When $\phi$ is unknown it is estimated *after* convergence, typically by
the Pearson statistic $\widehat\phi = \tfrac{1}{n-p}\sum_i (y_i-\widehat\mu_i)^2/V(\widehat\mu_i)$.

**Deviance.** Convergence and goodness-of-fit are monitored by the **deviance**
$D = 2\big[\ell(\text{saturated}) - \ell(\widehat\beta)\big]$, which IRLS decreases monotonically (with
step-halving) and which generalizes the residual sum of squares.

## Mapping to framework

- **Input:** $X\in\mathbb{R}^{n\times p}$, $y\in\mathbb{R}^n$, link $g$ and family ($V$, $\phi$).
- **Output:** point estimate $\widehat\beta$ (the MLE), plus $\widehat\phi$ and deviance $D$.
- **Links:** any differentiable $g$; identity/Gaussian reduces to a single
  [OLS](ols.md) solve, logit→logistic regression, log→Poisson regression.
- **Preprocessing:** intercept as a column of ones; standardization optional (improves conditioning
  of $X^\top W X$).

## Complexity

- Per IRLS iteration: $O(np^2)$ to form $X^\top W X$ and $O(p^3)$ to factor/solve; memory $O(np)$.
- Total: (#iterations) $\times\,O(np^2)$; typically 5–25 iterations thanks to local quadratic
  convergence.

## Statistical guarantees

- **Consistency & asymptotic normality.** Under standard regularity (correct family/link, full-rank
  design, $n\to\infty$ with $p$ fixed), $\widehat\beta$ is consistent and
  $\sqrt n(\widehat\beta-\beta^\star)\xrightarrow{d}\mathcal N\!\big(0,\,\phi\,(\tfrac1n X^\top W X)^{-1}\big)$,
  attaining the Cramér–Rao bound (efficiency).
- **Inference.** $\widehat{\operatorname{Var}}(\widehat\beta)=\widehat\phi\,(X^\top W X)^{-1}$ at convergence
  gives Wald standard errors; deviance differences give likelihood-ratio tests.
- Convergence is guaranteed locally; step-halving safeguards the global iteration.

## Variants & related

- [OLS](ols.md) — the single-step IRLS solution for Gaussian/identity.
- [Ridge Regression](ridge.md) and [Lasso via Coordinate Descent](lasso-cd.md) — penalized GLMs add
  a term to each IRLS step (penalized IRLS).
- **Quasi-likelihood / GEE** — replace the full likelihood by a mean–variance specification.

## References

- <a id="ref-McCullagh1989"></a> <a href="https://doi.org/10.1201/9780203753736" class="ref-link" target="_blank" rel="noopener noreferrer">McCullagh, P. and Nelder, J. A. (1989). *Generalized Linear Models*, 2nd ed. Chapman and Hall.</a>
- <a id="ref-IWLS1987"></a> <a href="https://www.jstor.org/stable/1403404" class="ref-link" target="_blank" rel="noopener noreferrer">Green, P. J. (1987). Penalized likelihood for general semi-parametric regression models. *Int. Stat. Rev.*, 55:245--259.</a>
- <a id="ref-Nocedal2006"></a> <a href="https://link.springer.com/book/10.1007/978-0-387-40065-5" class="ref-link" target="_blank" rel="noopener noreferrer">Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer.</a>
