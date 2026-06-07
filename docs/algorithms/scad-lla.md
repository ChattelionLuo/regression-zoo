---
id: scad-lla
name: "SCAD via Local Linear Approximation"
aliases: ["SCAD-LLA", "one-step SCAD", "nonconcave penalized likelihood (SCAD)"]
family: penalized-batch
regime: [high-dim, low-dim]
penalty: scad
link_support: [identity, logit, log]
output: path
year: 2008
refs: [fan2001variable, loh2015regularized]
status: draft
---

# SCAD via Local Linear Approximation

!!! info "At a glance"
    **Family:** penalized-batch (nonconvex) · **Regime:** high-dim / low-dim · **Penalty:** scad ·
    **Output:** path over $\lambda$ · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** fan2001variable, loh2015regularized

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family);
  Gaussian/identity is the canonical case, logistic/Poisson handled through the IRLS-weighted
  inner solver below.
- High- or low-dimensional; sparsity $\lVert\beta^\star\rVert_0=s\ll p$ is the motivating regime.
- Columns of $X$ standardized to unit variance; $y$ centered (Gaussian). Intercept unpenalized.
- The penalty is **nonconvex**: the goal is (near-)unbiased estimation of large coefficients
  while retaining sparsity, so the objective may have multiple stationary points.

## Estimator / objective

SCAD (Fan & Li 2001) uses a folded-concave penalty $p_\lambda(\cdot)$ applied coordinatewise:

$$
\hat\beta(\lambda) \;\in\; \arg\min_{\beta\in\mathbb{R}^p}\;
\mathcal L(\beta) \;+\; \sum_{j=1}^p p_\lambda\!\big(|\beta_j|\big),
$$

where $\mathcal L$ is the mean negative log-likelihood (Gaussian: $\tfrac1{2n}\lVert y-X\beta\rVert_2^2$).
The SCAD penalty is defined through its derivative, for $a>2$ (default $a=3.7$):

$$
p_\lambda'(t) \;=\; \lambda\left\{ \mathbf 1(t\le\lambda) \;+\;
\frac{(a\lambda - t)_+}{(a-1)\lambda}\,\mathbf 1(t>\lambda) \right\}, \qquad t\ge 0 .
$$

Integrating gives the penalty itself ($p_\lambda(0)=0$):

$$
p_\lambda(t)=
\begin{cases}
\lambda t, & 0\le t\le\lambda,\\[2pt]
\dfrac{2a\lambda t - t^2 - \lambda^2}{2(a-1)}, & \lambda<t\le a\lambda,\\[4pt]
\dfrac{(a+1)\lambda^2}{2}, & t> a\lambda.
\end{cases}
$$

The penalty is linear (lasso-like) near $0$, transitions quadratically, and is **flat**
($p_\lambda'\equiv 0$) for $t>a\lambda$ — large coefficients are left unpenalized, removing the
lasso's bias.

## Algorithm

**Local Linear Approximation (LLA, Zou & Li 2008).** Because $p_\lambda$ is concave on
$[0,\infty)$, majorize it at the current iterate $\beta^{(k)}$ by its tangent line,

$$
p_\lambda(|\beta_j|) \;\le\; p_\lambda(|\beta_j^{(k)}|) + p_\lambda'\!\big(|\beta_j^{(k)}|\big)\,
\big(|\beta_j| - |\beta_j^{(k)}|\big),
$$

which turns each step into a **weighted lasso** with weights
$w_j^{(k)}=p_\lambda'(|\beta_j^{(k)}|)$:

$$
\beta^{(k+1)} \;=\; \arg\min_{\beta}\;
\mathcal L(\beta) \;+\; \sum_{j=1}^p w_j^{(k)}\,|\beta_j| .
$$

```text
Input: X (standardized), y, λ, a (=3.7), init β^(0) (e.g. lasso solution)
for k = 0, 1, 2, ... until convergence:
    # 1. update weights from SCAD derivative
    for j = 1..p:
        w_j = λ * ( 1(|β_j^(k)| <= λ) + (aλ - |β_j^(k)|)_+ / ((a-1)λ) * 1(|β_j^(k)| > λ) )
    # 2. solve the weighted lasso (e.g. coordinate descent / IRLS inner loop)
    β^(k+1) = argmin_β  L(β) + Σ_j w_j |β_j|
return β^(λ)
```

- **One-step LLA.** Initialized at a good estimator (e.g. the lasso, or for $n>p$ the MLE),
  a *single* LLA iteration already attains the oracle property; further iterations refine it.
- **LQA alternative.** Fan & Li's original **Local Quadratic Approximation** majorizes
  $p_\lambda(|\beta_j|)\approx p_\lambda(|\beta_j^{(k)}|)+\tfrac12\frac{p_\lambda'(|\beta_j^{(k)}|)}{|\beta_j^{(k)}|}(\beta_j^2-\beta_j^{(k)2})$,
  yielding a ridge-like reweighting; it cannot set coefficients exactly to zero (requires
  thresholding) and is numerically less stable than LLA near $0$.
- **General GLM.** Each weighted-lasso subproblem is solved by penalized IRLS (outer quadratic
  approximation of $\mathcal L$) + weighted coordinate descent (inner), exactly as in
  [Lasso-CD](lasso-cd.md).

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ | path | selected by CV or BIC; controls sparsity |
| $a$ | $3.7$ | concavity / transition width; $a>2$ required. $3.7$ from Fan & Li (Bayes-risk argument) |
| init $\beta^{(0)}$ | lasso | warm start; quality matters for nonconvex landscape |
| LLA iterations | 1–3 | one-step often suffices; iterate to a fixed point for full convergence |
| standardize | true | columns to unit variance; coefficients returned on original scale |
| intercept | true, unpenalized | |
| inner solver | coordinate descent | weighted lasso per LLA step; IRLS outer loop for GLMs |

## Mapping to framework

- **Input:** $X, y$, link; regularization $\lambda$, concavity $a$ (or request full path).
- **Output:** $\hat\beta(\lambda)$ — a point or the whole path.
- **Links:** identity (LS inner loop), logit, log (IRLS outer loop).
- **Preprocessing:** standardize $X$; center $y$ (Gaussian) or fit an unpenalized intercept (GLM).

## Complexity

- Per LLA step = one weighted lasso fit: $O(npL)$-ish via warm-started coordinate descent.
- Total: (number of LLA iterations) $\times$ cost of a weighted lasso. One-step LLA is therefore
  only a constant factor above a single lasso solve.
- Memory $O(np)$ (or $O(\text{nnz})$ for sparse $X$).

## Statistical guarantees

- **Oracle property (Fan & Li 2001).** With a suitable $\lambda\to0$ rate, the SCAD estimator
  performs as well as if the true support were known: it selects the correct sparsity pattern
  with probability $\to 1$ and is asymptotically normal/efficient on the active set, with **no
  asymptotic bias** for large coefficients.
- **LLA recovers the oracle (Zou & Li 2008).** From a $\sqrt{n}$- (or lasso-) consistent
  initializer, one LLA step yields the oracle estimator with high probability.
- **Nonconvex M-estimation theory.** Under restricted strong convexity, all stationary points
  of the nonconvex objective lie within statistical precision of $\beta^\star$
  (`loh2015regularized`), so the LLA fixed point is well-behaved despite nonconvexity.

## Variants & related

- [MCP via Coordinate Descent](mcp-cd.md) — sibling folded-concave penalty.
- [Lasso via Coordinate Descent](lasso-cd.md) — the convex special case / inner solver.
- [Adaptive Lasso](adaptive-lasso.md) — also a reweighted $\ell_1$ method with the oracle property.

## References

- Fan & Li (2001), *Variable selection via nonconcave penalized likelihood and its oracle
  properties* (`fan2001variable`) — defines SCAD and LQA; establishes the oracle property.
- Zou & Li (2008), *One-step sparse estimates in nonconcave penalized likelihood models*
  (*Annals of Statistics*) — introduces the LLA algorithm and one-step estimator (not in
  `reference.bib`; named here in prose).
- Loh & Wainwright (2015), *Regularized M-estimators with nonconvexity* (`loh2015regularized`)
  — statistical control of stationary points under nonconvex penalties.
