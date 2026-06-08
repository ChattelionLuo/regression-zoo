---
id: fista
name: "FISTA (Fast ISTA)"
aliases: ["accelerated proximal gradient", "Nesterov-accelerated ISTA", "fast iterative shrinkage-thresholding"]
family: first-order-prox
regime: [high-dim, low-dim]
penalty: lasso
link_support: [identity, logit, log]
output: point
year: 2009
refs: [beck2009fast, daubechies2004iterative]
status: draft
---

# FISTA (Fast ISTA)

!!! info "At a glance"
    **Family:** first-order-prox · **Regime:** high-dim / low-dim · **Penalty:** lasso (any prox-able) ·
    **Output:** point · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** beck2009fast, daubechies2004iterative

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family)
  whose loss $\mathcal L$ is convex and differentiable with **Lipschitz-continuous gradient**
  (constant $L_\nabla$); Gaussian/identity is the canonical case.
- High- or low-dimensional. The penalty $P$ need only admit a cheap proximal operator
  (lasso is the headline case; any prox-able $P$ works).
- Columns of $X$ standardized; $y$ centered (Gaussian). Intercept unpenalized.

## Estimator / objective

FISTA solves the same composite convex objective as [ISTA](ista.md):

$$
\hat\beta(\lambda) \;=\; \arg\min_{\beta\in\mathbb{R}^p}\;
\mathcal L(\beta) \;+\; \lambda\lVert\beta\rVert_1 ,
\qquad
\mathcal L(\beta)=\frac{1}{2n}\lVert y-X\beta\rVert_2^2 \;\text{(Gaussian)} ,
$$

but converges faster. For a general GLM $\mathcal L$ is the mean negative log-likelihood.

## Algorithm

**Accelerated proximal gradient (Beck & Teboulle 2009).** Apply the proximal-gradient step at an
**extrapolated point** $y^{(k)}$ that carries Nesterov momentum, using a fixed step $s=1/L_\nabla$.
Initialize $\beta^{(0)}=\beta^{(-1)}$, $t_1=1$. The exact recursion is

$$
\beta^{(k+1)} = \operatorname{prox}_{s\lambda\lVert\cdot\rVert_1}\!\Big(y^{(k+1)} - s\,\nabla\mathcal L(y^{(k+1)})\Big),
$$
$$
t_{k+1} = \frac{1+\sqrt{1+4t_k^2}}{2},
\qquad
y^{(k+1)} = \beta^{(k)} + \frac{t_k - 1}{t_{k+1}}\big(\beta^{(k)} - \beta^{(k-1)}\big).
$$

```text
Input: X, y, λ, gradient ∇L, step s = 1/L_∇, init β^(0)
β^(-1) = β^(0);  y^(1) = β^(0);  t_1 = 1
for k = 1, 2, ... until convergence:
    v        = y^(k) - s * ∇L(y^(k))            # gradient step at extrapolated point
    β^(k)    = Soft(v, s * λ)                    # prox: soft-threshold componentwise
    t_{k+1}  = (1 + sqrt(1 + 4 t_k^2)) / 2       # momentum weight
    y^(k+1)  = β^(k) + ((t_k - 1)/t_{k+1}) (β^(k) - β^(k-1))   # extrapolation
return β̂
```

- The momentum term $\tfrac{t_k-1}{t_{k+1}}\to 1$ as $k\to\infty$, adding a vanishing fraction of
  the previous step's direction — this is what lifts the rate from $O(1/k)$ to $O(1/k^2)$.
- **Backtracking variant.** When $L_\nabla$ is unknown, search $s_k\leftarrow\eta s_k$ until the
  majorization inequality holds at $y^{(k)}$, and use $s_k$ in both the prox level and the
  $t$-update; convergence guarantees are preserved.
- **Stopping.** Relative change in objective or iterate below tolerance.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ | fixed / path | regularization strength; path via warm starts |
| step $s$ | $1/L_\nabla$ | constant ($L_\nabla$ = Lipschitz const of $\nabla\mathcal L$) or backtracking |
| backtracking $\eta$ | $0.5$ | step-shrink factor when $L_\nabla$ unknown |
| momentum | Nesterov $t_k$ | fixed recursion above; no tuning needed |
| tol | $10^{-6}$ | stopping on objective / iterate change |
| max iter | $10^2$–$10^3$ | far fewer than ISTA thanks to $O(1/k^2)$ rate |

## Mapping to framework

- **Input:** $X, y$, link; regularization $\lambda$; step rule.
- **Output:** $\hat\beta(\lambda)$ — a single point.
- **Links:** identity, logit, log (any convex GLM loss with Lipschitz gradient).
- **Preprocessing:** standardize $X$; center $y$ (Gaussian) or fit an unpenalized intercept (GLM).

## Complexity

- Per iteration: one gradient ($O(np)$ dense) + one prox ($O(p)$) + $O(p)$ extrapolation.
  Same per-step cost as ISTA. Memory $O(np)$ (plus $O(p)$ to store $\beta^{(k-1)}$).
- Iteration count: $O(1/\sqrt{\epsilon})$ for an $\epsilon$-accurate objective — quadratically
  fewer iterations than ISTA's $O(1/\epsilon)$.

## Statistical guarantees

- **Optimization (convex $\mathcal L$).** With $s=1/L_\nabla$,
  $F(\beta^{(k)})-F(\hat\beta)\le \dfrac{2L_\nabla\lVert\beta^{(0)}-\hat\beta\rVert_2^2}{(k+1)^2}=O(1/k^2)$,
  where $F=\mathcal L+\lambda P$ — the optimal rate for first-order methods on this class
  [[Beck and Teboulle, 2009]](#ref-beck2009fast).
- FISTA is an optimization method: statistical properties of the *solution* $\hat\beta(\lambda)$
  are those of the lasso/penalized M-estimator it computes (see [Lasso-CD](lasso-cd.md)).
- Builds on the iterative thresholding iteration of Daubechies, Defrise & De Mol (2004)
  [[Daubechies et al., 2004]](#ref-daubechies2004iterative).

## Variants & related

- [ISTA (Iterative Shrinkage-Thresholding)](ista.md) — the unaccelerated $O(1/k)$ base method.
- [Lasso via Coordinate Descent](lasso-cd.md) — alternative solver for the same objective.
- **Accelerated proximal gradient with general $P$** — group lasso, nuclear norm, etc.

## References

- <a id="ref-beck2009fast"></a> Beck, A. and Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM J. Imaging Sci.*, 2(1):183--202.
- <a id="ref-daubechies2004iterative"></a> Daubechies, I., Defrise, M., and De Mol, C. (2004). An iterative thresholding algorithm for linear inverse problems with a sparsity constraint. *Commun. Pure Appl. Math.*, 57(11):1413--1457.
