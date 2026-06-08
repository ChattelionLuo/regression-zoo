---
id: mcp-cd
name: "MCP via Coordinate Descent"
aliases: ["minimax concave penalty", "MCP-CD", "MC+", "ncvreg MCP"]
family: penalized-batch
regime: [high-dim, low-dim]
penalty: mcp
link_support: [identity, logit, log]
output: path
year: 2010
refs: [zhang2010nearly, loh2015regularized]
status: draft
---

# MCP via Coordinate Descent

!!! info "At a glance"
    **Family:** penalized-batch (nonconvex) · **Regime:** high-dim / low-dim · **Penalty:** mcp ·
    **Output:** path over $\lambda$ · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** zhang2010nearly, loh2015regularized

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family);
  Gaussian/identity is the canonical case, logistic/Poisson handled through the IRLS-weighted
  inner solver below.
- High- or low-dimensional; sparsity $\lVert\beta^\star\rVert_0=s\ll p$ is the motivating regime.
- Columns of $X$ standardized so that $\tfrac1n\lVert X_{\cdot j}\rVert_2^2=1$; $y$ centered
  (Gaussian). Intercept unpenalized.
- The penalty is **nonconvex** but, paired with a strongly convex loss, the *overall* objective
  can remain convex (see condition below).

## Estimator / objective

The Minimax Concave Penalty (Zhang 2010) is applied coordinatewise:

$$
\hat\beta(\lambda) \;\in\; \arg\min_{\beta\in\mathbb{R}^p}\;
\frac{1}{2n}\lVert y - X\beta\rVert_2^2 \;+\; \sum_{j=1}^p p_\lambda\!\big(|\beta_j|;\gamma\big),
$$

with MCP defined through its derivative, for $\gamma>1$:

$$
p_\lambda'(t;\gamma) \;=\; \Big(\lambda - \frac{t}{\gamma}\Big)_+, \qquad t\ge 0 .
$$

Integrating ($p_\lambda(0)=0$) gives

$$
p_\lambda(t;\gamma)=
\begin{cases}
\lambda t - \dfrac{t^2}{2\gamma}, & 0\le t\le \gamma\lambda,\\[6pt]
\dfrac{\gamma\lambda^2}{2}, & t> \gamma\lambda.
\end{cases}
$$

Like SCAD, MCP starts with the lasso slope $\lambda$ at the origin and relaxes the penalization
rate continuously to $0$, becoming flat for $t>\gamma\lambda$ (no bias on large coefficients).
For a GLM, replace the squared-error loss with the mean negative log-likelihood $\mathcal L(\beta)$.

## Algorithm

**Cyclic coordinate descent with firm thresholding.** With standardized columns, let the partial
residual be $r^{(j)} = y - \sum_{k\ne j} X_{\cdot k}\beta_k$ and
$z_j = \tfrac1n X_{\cdot j}^\top r^{(j)}$. The MCP coordinate update is the **firm-thresholding**
operator $F(\cdot;\lambda,\gamma)$:

$$
\beta_j \;\leftarrow\; F(z_j;\lambda,\gamma)=
\begin{cases}
\dfrac{\mathcal S_{\lambda}(z_j)}{\,1 - 1/\gamma\,}, & |z_j|\le \gamma\lambda,\\[8pt]
z_j, & |z_j| > \gamma\lambda,
\end{cases}
\qquad
\mathcal S_{\lambda}(z)=\operatorname{sign}(z)\,(|z|-\lambda)_+ .
$$

```text
Input: X (standardized), y (centered), λ-grid λ_max>...>λ_min, γ (>1)
Warm starts along the grid (pathwise):
for λ in grid:                       # decreasing
    repeat until convergence:
        for j = 1..p:
            r   = y - X β + X[:,j] β_j           # partial residual
            z_j = (1/n) X[:,j]ᵀ r
            if |z_j| <= γλ:
                β_j = Soft(z_j, λ) / (1 - 1/γ)   # firm threshold
            else:
                β_j = z_j
    record β(λ)
return path {β(λ)}
```

- $\lambda_{\max}=\tfrac1n\lVert X^\top y\rVert_\infty$ (smallest $\lambda$ with $\hat\beta=0$);
  grid log-spaced down to $\lambda_{\min}=\epsilon\lambda_{\max}$, with warm starts.
- **General GLM.** Penalized IRLS (outer quadratic approximation) + weighted coordinate descent
  (inner) applying the firm-thresholding update on the working response, as in [Lasso-CD](lasso-cd.md).

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ | path | selected by CV or BIC |
| $\gamma$ | $3$ | concavity; $\gamma>1$ required. $\gamma\to\infty$ → lasso (soft), $\gamma\to1^+$ → hard threshold |
| grid length | 100 | log-spaced $\lambda_{\max}\to\epsilon\lambda_{\max}$ |
| standardize | true | columns to unit variance; coefficients returned on original scale |
| intercept | true, unpenalized | |
| tol | $10^{-7}$ | convergence on max coordinate change |
| family/link | gaussian/identity | also binomial/logit, poisson/log via IRLS |

## Mapping to framework

- **Input:** $X, y$, link; regularization $\lambda$, concavity $\gamma$ (or request full path).
- **Output:** $\hat\beta(\lambda)$ — a point or the whole path.
- **Links:** identity (LS inner loop), logit, log (IRLS outer loop).
- **Preprocessing:** standardize $X$; center $y$ (Gaussian) or fit an unpenalized intercept (GLM).

## Complexity

- Per full cycle: $O(np)$ (Gaussian, dense), or $O(n\,\lvert\text{active set}\rvert)$ with active-set tricks.
- Whole path of $L$ values with warm starts: typically near $O(npL)$ in practice.
- Memory $O(np)$ (or $O(\text{nnz})$ for sparse $X$).

## Statistical guarantees

- **Convexity condition.** For Gaussian loss with standardized columns, the objective is convex
  whenever $\gamma > 1/c_{\min}$, where $c_{\min}$ is the minimum eigenvalue of $\tfrac1n X^\top X$;
  more generally each coordinate subproblem is convex when $\gamma>1$. This makes MCP-CD better
  behaved than generic nonconvex problems.
- **Near-unbiased selection (Zhang 2010).** Under a sparse Riesz / restricted-eigenvalue
  condition, MCP attains the oracle sparsity pattern and near-minimax estimation rates while
  minimizing the maximum concavity for a given threshold gap [[Zhang, 2010]](#ref-zhang2010nearly).
- **Stationary-point control.** As a regularized M-estimator with nonconvex penalty, all
  stationary points are within statistical precision of $\beta^\star$ under restricted strong
  convexity [[Loh and Wainwright, 2015]](#ref-loh2015regularized).

## Variants & related

- [SCAD via Local Linear Approximation](scad-lla.md) — sibling folded-concave penalty.
- [Lasso via Coordinate Descent](lasso-cd.md) — convex limit ($\gamma\to\infty$) / inner solver.
- **Hard thresholding** — the $\gamma\to1^+$ limit of the firm-thresholding operator.

## References

- <a id="ref-zhang2010nearly"></a> Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty. *Ann. Statist.*, 38(2):894--942.
- <a id="ref-loh2015regularized"></a> Loh, P.-L. and Wainwright, M. J. (2015). Regularized M-estimators with nonconvexity: statistical and algorithmic theory for local optima. *J. Mach. Learn. Res.*, 16:559--616.
