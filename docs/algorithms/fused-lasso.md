---
id: fused-lasso
name: "Fused Lasso"
aliases: ["fused lasso signal approximator", "FLSA", "total-variation regression"]
family: penalized-batch
regime: [low-dim, high-dim]
penalty: fused-lasso
link_support: [identity]
output: path
year: 2005
refs: [Tibshirani2005, tibshirani1996regression]
status: draft
---

# Fused Lasso

!!! info "At a glance"
    **Family:** penalized-batch · **Regime:** low-dim / high-dim · **Penalty:** fused-lasso ·
    **Output:** point / path · **Links:** identity · **Status:** draft ·
    **Refs:** Tibshirani2005, tibshirani1996regression

## Setting & assumptions

- Gaussian family with **identity link** ($y=X\beta^\star+\varepsilon$). The fused penalty is
  defined relative to a **meaningful ordering of the coefficients** $\beta_1,\dots,\beta_p$
  (e.g. genomic position, time, a 1-D index) along which neighbouring coefficients are expected
  to be similar.
- Targets solutions that are simultaneously **sparse** and **piecewise constant** over that
  ordering — many coefficients exactly zero, and many adjacent coefficients exactly equal.
- Low- or high-dimensional; the special case $X=I$ (the *fused lasso signal approximator*, FLSA)
  is the canonical denoising/segmentation problem.
- Columns standardized; $y$ centered. Intercept unpenalized.

## Estimator / objective

Two $\ell_1$ penalties are combined: one on the coefficients (sparsity) and one on **successive
differences** (smoothness / piecewise constancy):

$$
\hat\beta \;=\; \arg\min_{\beta\in\mathbb{R}^p}\;
\frac{1}{2n}\lVert y-X\beta\rVert_2^2
\;+\; \lambda_1 \lVert\beta\rVert_1
\;+\; \lambda_2 \sum_{j=2}^{p} |\beta_j-\beta_{j-1}| .
$$

The second term is the total variation of $\beta$ along the ordering; it shrinks adjacent
differences to **exactly zero**, producing flat segments. Setting $\lambda_2=0$ recovers the
[lasso](lasso-cd.md); setting $\lambda_1=0$ gives 1-D total-variation denoising. The objective is
convex but the difference penalty is **not separable** across coordinates, which dictates the
solvers below.

## Algorithm

Because of the coupling between adjacent coefficients, **plain coordinate descent does not
generally converge to the optimum** — minimizing one $\beta_j$ at a time can stall at a non-stationary
point where two fused coordinates would need to move together (Arena note: this non-separability
is the defining computational difference from the lasso/elastic net). Faithful solvers instead are:

**1. Original two-phase / SQP solver (Tibshirani et al., 2005).** Reformulate as a quadratic
program by splitting each $\beta_j=\beta_j^+-\beta_j^-$ and introducing variables for the
differences, turning both $\ell_1$ terms into linear constraints, then solve with a
sequential-quadratic-programming / two-phase active-set method that adds and drops the
$|\beta_j|=0$ and $|\beta_j-\beta_{j-1}|=0$ constraints.

**2. FLSA special case $X=I$ — solved exactly.** The 1-D signal-approximator

$$
\min_{\beta}\;\tfrac12\lVert y-\beta\rVert_2^2 + \lambda_1\lVert\beta\rVert_1
   + \lambda_2\sum_{j\ge 2}|\beta_j-\beta_{j-1}|
$$

is solved **exactly** and in (near-)linear time. First solve the pure total-variation problem
($\lambda_1=0$) by the **taut-string** method or an $O(p)$ **dynamic program**, then apply
soft-thresholding at level $\lambda_1$ to the result (the two operations commute for FLSA).

```text
Input: y, ordering 1..p, λ1 (sparsity), λ2 (fusion)
General X:
    Solve the QP   min (1/2n)||y - Xβ||² + λ1 Σ|β_j| + λ2 Σ|β_j − β_{j-1}|
    via the two-phase / SQP active-set method (Tibshirani et al. 2005).
    NOTE: coordinate descent does NOT generally work — the difference penalty
          is non-separable; fused coordinates must move jointly.

Special case X = I (FLSA), exact and fast:
    1. θ = TV_denoise(y, λ2)        # taut-string or O(p) dynamic program (λ1 = 0)
    2. β = Soft(θ, λ1)             # soft-threshold each coordinate at λ1
    Return β
```

A path over $(\lambda_1,\lambda_2)$ is traced by warm-starting across a 2-D grid, or via
path/homotopy algorithms for the generalized-lasso form $\lVert D\beta\rVert_1$ with the
difference operator $D$.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda_1$ (sparsity) | tuned | $\ell_1$ on coefficients; CV / AIC / BIC |
| $\lambda_2$ (fusion) | tuned | $\ell_1$ on successive differences; controls segment count |
| ordering | user-supplied | the 1-D order defining neighbours — a required input |
| standardize | true | columns to unit variance |
| intercept | true, unpenalized | |
| solver | SQP (general $X$), taut-string/DP ($X=I$) | |

The two penalties are typically selected jointly by 2-D cross-validation.

## Mapping to framework

- **Input:** $X, y$, link `identity`, an **ordering** of coefficients; penalties $\lambda_1,\lambda_2$.
- **Output:** $\hat\beta$ — a point, or a path over the $(\lambda_1,\lambda_2)$ grid; sparse and
  piecewise constant.
- **Links:** identity only (for non-Gaussian families a fused penalty is combined with IRLS, but
  that is outside the original definition).
- **Preprocessing:** standardize $X$; center $y$; supply the neighbour ordering.

## Complexity

- FLSA ($X=I$): $O(p)$ (dynamic program / taut string) plus $O(p)$ soft-threshold — essentially linear.
- General $X$: dominated by the QP/SQP solver; cost depends on the active-set size per step and is
  super-linear in $p$ in the worst case. Generalized-lasso path solvers cost
  $O(\,\text{breakpoints}\times p^2\,)$-type effort.
- Memory $O(np)$ (or $O(p)$ for FLSA).

## Statistical guarantees

- For 1-D total-variation denoising the fused estimator is **minimax-rate optimal** for signals of
  bounded variation / with few jumps, and recovers the change-point locations consistently under a
  minimum-jump-size condition.
- In regression, support and "fusion" recovery hold under design conditions analogous to the
  irrepresentable condition applied to $(\,\beta,\ D\beta\,)$ jointly.

## Variants & related

- [Lasso](lasso-cd.md) — the $\lambda_2=0$ special case.
- **1-D total variation / taut string** — the $\lambda_1=0$, $X=I$ special case.
- **Generalized lasso** $\lVert D\beta\rVert_1$ — fused lasso is the choice of $D=$ first-difference
  (plus identity) operator; trend filtering uses higher-order differences.
- [Group Lasso](group-lasso.md) · [Elastic Net](elastic-net.md) — other structured penalties.

## References

- <a id="ref-Tibshirani2005"></a> Tibshirani, R., Saunders, M., Rosset, S., Zhu, J., and Knight, K. (2005). Sparsity and smoothness via the fused lasso. *J. R. Stat. Soc. Ser. B*, 67(1):91--108.
- <a id="ref-tibshirani1996regression"></a> Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *J. R. Stat. Soc. Ser. B*, 58(1):267--288.
