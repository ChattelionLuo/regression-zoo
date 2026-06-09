---
id: group-lasso
name: "Group Lasso"
aliases: ["grouped lasso", "group L1/L2", "Yuan-Lin group lasso"]
family: penalized-batch
regime: [high-dim, low-dim]
penalty: group-lasso
link_support: [identity, logit, log]
output: path
year: 2006
refs: [SGL2015, buhlmann2011statistics, fhht2007]
status: draft
---

# Group Lasso

!!! info "At a glance"
    **Family:** penalized-batch · **Regime:** high-dim / low-dim · **Penalty:** group-lasso ·
    **Output:** path over $\lambda$ · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** [`Buhlmann and van de Geer, 2011`](#ref-buhlmann2011statistics)

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family);
  Gaussian/identity is canonical, logistic/Poisson via the IRLS outer loop.
- Predictors carry a **known partition** into $G$ groups
  $\{1,\dots,p\}=\bigcup_{g=1}^G \mathcal G_g$ with blocks $\beta_g\in\mathbb{R}^{d_g}$,
  $d_g=|\mathcal G_g|$, and group submatrices $X_g\in\mathbb{R}^{n\times d_g}$. Groups arise from
  factor dummy variables, basis expansions, or multi-task structure.
- High- or low-dimensional; the target is **groupwise sparsity** — whole groups are in or out.
- Columns standardized; $y$ centered (Gaussian). Intercept unpenalized. Within-group orthonormality
  ($\tfrac1n X_g^\top X_g = I_{d_g}$) gives the cleanest update and is often arranged by a
  per-group orthogonalization.

## Estimator / objective

The penalty is a sum of (un-squared) $\ell_2$ norms of the blocks, weighted by $\sqrt{d_g}$ so that
larger groups are penalized commensurately:

$$
\widehat\beta(\lambda) \;=\; \arg\min_{\beta\in\mathbb{R}^p}\;
\frac{1}{2n}\Big\lVert y-\sum_{g=1}^G X_g\beta_g\Big\rVert_2^2
\;+\; \lambda \sum_{g=1}^G \sqrt{d_g}\,\lVert\beta_g\rVert_2 .
$$

For a general GLM, replace the Gaussian loss by the mean negative log-likelihood $\mathcal L(\beta)$.
Because $\lVert\beta_g\rVert_2$ (not $\lVert\beta_g\rVert_2^2$) is non-differentiable exactly at
$\beta_g=0$, the penalty drives **entire blocks** to zero. With all $d_g=1$ it reduces to the
ordinary [lasso](lasso-cd.md).

## Algorithm

**Block coordinate descent.** The penalty is separable *across groups*, so we cyclically minimize
over one block $\beta_g$ at a time. Let the group partial residual be
$r_g = y - \sum_{h\ne g} X_h\beta_h$ and $z_g = \tfrac1n X_g^\top r_g$. The block KKT condition
gives a **groupwise soft-threshold**: the optimal $\beta_g=0$ iff
$\lVert z_g\rVert_2 \le \lambda\sqrt{d_g}$. Under within-group orthonormality
($\tfrac1n X_g^\top X_g=I$) the nonzero update is in closed form,

$$
\beta_g \;\leftarrow\;
\Big(1 - \frac{\lambda\sqrt{d_g}}{\lVert z_g\rVert_2}\Big)_{\!+}\; z_g ,
\qquad
z_g = \tfrac1n X_g^\top r_g ,
$$

a vector soft-threshold that shrinks the block toward $0$ along its own direction (and zeros it
entirely when $\lVert z_g\rVert_2\le\lambda\sqrt{d_g}$).

```text
Input: X partitioned into groups {X_g}, y (centered), λ-grid, sizes d_g
Warm starts along the grid (pathwise):
for λ in grid:                              # decreasing
    repeat until convergence:
        for g = 1..G:
            r_g = y - X β + X_g β_g          # group partial residual
            z_g = (1/n) X_gᵀ r_g
            if ||z_g||_2 <= λ·sqrt(d_g):
                β_g = 0                      # whole group out
            else:
                β_g = (1 - λ·sqrt(d_g)/||z_g||_2) · z_g     # orthonormal X_g
                # general X_g: solve the group subproblem by a 1-D fixed-point /
                # Newton iteration on the block, or a few prox-gradient steps
    record β(λ)
Return path {β(λ)}
```

- $\lambda_{\max}=\max_g \tfrac{1}{n\sqrt{d_g}}\lVert X_g^\top y\rVert_2$ (smallest $\lambda$ with
  $\widehat\beta=0$); grid log-spaced to $\epsilon\lambda_{\max}$.
- For **non-orthonormal** $X_g$ the inner block problem has no closed form; solve it with a short
  proximal-gradient / fixed-point loop (the threshold test for $\beta_g=0$ is unchanged).

**General GLM — penalized IRLS (outer) + block coordinate descent (inner)**, exactly as for the
lasso but with the groupwise update replacing the scalar one.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ | path | selected by CV (`lambda.min` / `lambda.1se`), AIC/BIC, or fixed |
| group weights | $\sqrt{d_g}$ | the standard size-adjusted weighting; can be overridden |
| grouping | user-supplied | the known partition $\{\mathcal G_g\}$ — a required input |
| standardize | true | per-group orthonormalization recommended for the closed-form update |
| intercept | true, unpenalized | |
| tol | $10^{-7}$ | convergence on max block change |
| family/link | gaussian/identity | also binomial/logit, poisson/log via IRLS |

## Mapping to framework

- **Input:** $X, y$, link, the **group partition** $\{\mathcal G_g\}$; regularization $\lambda$.
- **Output:** $\widehat\beta(\lambda)$ — a point or the whole path; sparsity is at the group level.
- **Links:** identity (LS inner loop), logit, log (IRLS outer loop).
- **Preprocessing:** standardize $X$; optionally orthonormalize each $X_g$; center $y$ (Gaussian).

## Complexity

- Per full cycle: $O(np)$ for the residual/correlation products; the per-group closed form is
  $O(n d_g)$, and non-orthonormal blocks add the cost of the inner $d_g$-dimensional solve.
- Whole path of $L$ values with warm starts: near $O(npL)$ in practice.
- Memory $O(np)$ (or $O(\text{nnz})$ for sparse $X$).

## Statistical guarantees

- **Groupwise sparsity / selection.** Under a group-restricted-eigenvalue condition and
  $\lambda\asymp\sigma\big(\sqrt{d_g/n}+\sqrt{\log G/n}\big)$, the group lasso recovers the true
  active groups and attains $\ell_2$ error scaling with the total active dimension
  $\sum_{g\in S}d_g$ rather than $p$ [`Bühlmann and van de Geer, 2011`](#ref-buhlmann2011statistics).
- The $\sqrt{d_g}$ weighting equalizes the noise level across groups of differing size, which is
  what makes a single $\lambda$ appropriate for all groups.

## Variants & related

- [Lasso](lasso-cd.md) — the all-singletons case ($d_g\equiv 1$).
- **Sparse group lasso** — adds an extra $\ell_1$ term for within-group sparsity [`Simon et al., 2013`](#ref-simon2013sparse).
- [Elastic Net](elastic-net.md) · [Adaptive Lasso](adaptive-lasso.md) ·
  [Fused Lasso](fused-lasso.md) — sibling structured penalties.
- **Overlapping group lasso** — groups may share variables (latent-group formulation).

## References

- <a id="ref-Yuan2006group"></a> Yuan, M. and Lin, Y. (2006). Model selection and estimation in regression with grouped variables. *J. R. Stat. Soc. Ser. B*, 68(1):49--67.
- <a id="ref-simon2013sparse"></a> Simon, N., Friedman, J., Hastie, T., and Tibshirani, R. (2013). A sparse-group lasso. *J. Comput. Graph. Statist.*, 22(2):231--245.
- <a id="ref-buhlmann2011statistics"></a> Bühlmann, P. and van de Geer, S. (2011). *Statistics for High-Dimensional Data: Methods, Theory and Applications*. Springer.
- <a id="ref-fhht2007"></a> Friedman, J., Hastie, T., Höfling, H., and Tibshirani, R. (2007). Pathwise coordinate optimization. *Ann. Appl. Stat.*, 1(2):302--332.
