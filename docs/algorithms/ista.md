---
id: ista
name: "ISTA (Iterative Shrinkage-Thresholding)"
aliases: ["proximal gradient", "iterative soft-thresholding", "forward-backward splitting"]
family: first-order-prox
regime: [high-dim, low-dim]
penalty: lasso
link_support: [identity, logit, log]
output: point
year: 2004
refs: [daubechies2004iterative, beck2009fast]
status: draft
---

# ISTA (Iterative Shrinkage-Thresholding)

!!! info "At a glance"
    **Family:** first-order-prox · **Regime:** high-dim / low-dim · **Penalty:** lasso (any prox-able) ·
    **Output:** point · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** daubechies2004iterative, beck2009fast

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family)
  whose loss $\mathcal L$ is convex and differentiable with **Lipschitz-continuous gradient**
  (constant $L_\nabla$); Gaussian/identity is the canonical case.
- High- or low-dimensional. The penalty $P$ need only admit a cheap proximal operator
  (lasso is the headline case; the method extends to any prox-able $P$).
- Columns of $X$ standardized; $y$ centered (Gaussian). Intercept unpenalized.

## Estimator / objective

ISTA targets the composite convex objective

$$
\hat\beta(\lambda) \;=\; \arg\min_{\beta\in\mathbb{R}^p}\;
\mathcal L(\beta) \;+\; \lambda\lVert\beta\rVert_1 ,
\qquad
\mathcal L(\beta)=\frac{1}{2n}\lVert y-X\beta\rVert_2^2 \;\text{(Gaussian)} ,
$$

a smooth loss $\mathcal L$ plus a (possibly nonsmooth) penalty $\lambda P$. For a general GLM
$\mathcal L$ is the mean negative log-likelihood.

## Algorithm

**Proximal gradient.** Take a gradient step on the smooth part, then apply the proximal operator
of the penalty. With step size $t_k>0$,

$$
\beta^{(t+1)} \;=\; \operatorname{prox}_{t_k\lambda\lVert\cdot\rVert_1}\!\Big(\beta^{(t)} - t_k\,\nabla\mathcal L(\beta^{(t)})\Big)
\;=\; \mathcal S_{t_k\lambda}\!\Big(\beta^{(t)} - t_k\,\nabla\mathcal L(\beta^{(t)})\Big),
$$

where $\mathcal S$ is the soft-threshold (the lasso prox) applied componentwise.

```text
Input: X, y, λ, gradient ∇L, step rule, init β^(0)
for t = 0, 1, 2, ... until convergence:
    g       = ∇L(β^(t))                 # e.g. -(1/n) Xᵀ(y - Xβ^(t)) for Gaussian
    v       = β^(t) - t_k * g           # gradient step
    β^(t+1) = Soft(v, t_k * λ)          # prox: soft-threshold componentwise
return β̂
```

- **Constant step size.** $t_k = 1/L_\nabla$ guarantees descent and convergence. For Gaussian
  loss $L_\nabla = \tfrac1n\lambda_{\max}(X^\top X)$ (largest eigenvalue).
- **Backtracking line search.** When $L_\nabla$ is unknown, shrink $t_k\leftarrow\eta t_k$ (e.g.
  $\eta=0.5$) until the quadratic majorization
  $\mathcal L(\beta^{(t+1)})\le \mathcal L(\beta^{(t)})+\nabla\mathcal L(\beta^{(t)})^\top(\beta^{(t+1)}-\beta^{(t)})+\tfrac{1}{2t_k}\lVert\beta^{(t+1)}-\beta^{(t)}\rVert_2^2$
  holds.
- **Stopping.** Relative change in objective or $\lVert\beta^{(t+1)}-\beta^{(t)}\rVert$ below tolerance.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ | fixed / path | regularization strength; path via warm starts |
| step $t_k$ | $1/L_\nabla$ | constant ($L_\nabla$ = Lipschitz const of $\nabla\mathcal L$) or backtracking |
| backtracking $\eta$ | $0.5$ | step-shrink factor when $L_\nabla$ unknown |
| tol | $10^{-6}$ | stopping on objective / iterate change |
| max iter | $10^3$–$10^4$ | ISTA needs many iterations ($O(1/k)$ rate) |
| init $\beta^{(0)}$ | $0$ | warm start across a $\lambda$-path accelerates convergence |

## Mapping to framework

- **Input:** $X, y$, link; regularization $\lambda$; step rule.
- **Output:** $\hat\beta(\lambda)$ — a single point.
- **Links:** identity, logit, log (any convex GLM loss with Lipschitz gradient).
- **Preprocessing:** standardize $X$; center $y$ (Gaussian) or fit an unpenalized intercept (GLM).

## Complexity

- Per iteration: one gradient ($O(np)$ for dense $X$) + one prox ($O(p)$). Memory $O(np)$
  (or $O(\text{nnz})$ for sparse $X$).
- Iteration count: $O(1/\epsilon)$ iterations for an $\epsilon$-accurate objective (sublinear);
  linear rate if $\mathcal L$ is additionally strongly convex.

## Statistical guarantees

- **Optimization (convex $\mathcal L$).** With $t_k=1/L_\nabla$,
  $F(\beta^{(k)})-F(\hat\beta)\le \dfrac{L_\nabla\lVert\beta^{(0)}-\hat\beta\rVert_2^2}{2k}=O(1/k)$,
  where $F=\mathcal L+\lambda P$ (`beck2009fast`). The accelerated [FISTA](fista.md) improves this
  to $O(1/k^2)$.
- ISTA is an optimization method: statistical properties of the *solution* $\hat\beta(\lambda)$
  are those of the lasso/penalized M-estimator it computes (see [Lasso-CD](lasso-cd.md)).
- Daubechies, Defrise & De Mol (2004) established convergence of the iterative thresholding
  iteration for linear inverse problems with a sparsity penalty (`daubechies2004iterative`).

## Variants & related

- [FISTA (Fast ISTA)](fista.md) — Nesterov-accelerated version, $O(1/k^2)$.
- [Lasso via Coordinate Descent](lasso-cd.md) — alternative solver for the same objective.
- **Proximal gradient with general $P$** — group lasso, nuclear norm, etc., by swapping the prox.

## References

- Daubechies, Defrise & De Mol (2004), *An iterative thresholding algorithm for linear inverse
  problems with a sparsity constraint* (`daubechies2004iterative`).
- Beck & Teboulle (2009), *A fast iterative shrinkage-thresholding algorithm for linear inverse
  problems* (`beck2009fast`) — states ISTA, its $O(1/k)$ rate, and backtracking.
