---
id: truncated-gradient
name: "Truncated Gradient"
aliases: ["truncated gradient descent", "sparse online learning via truncated gradient"]
family: online-streaming
regime: [streaming, high-dim, low-dim]
penalty: lasso
link_support: [identity, logit, log]
output: point
year: 2009
refs: [truncated_SGD_Langford_2009]
status: draft
---

# Truncated Gradient

!!! info "At a glance"
    **Family:** online-streaming · **Regime:** streaming / high-dim / low-dim · **Penalty:** lasso ·
    **Output:** point · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** truncated_SGD_Langford_2009

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family)
  with canonical link; identity, logit, log are the primary cases.
- **Streaming / online**: points arrive sequentially; per-step cost and storage are bounded.
- $\ell_1$-type online sparsity is desired; naïve rounding of small SGD coefficients is too
  aggressive, so truncation is applied *gradually*.

## Estimator / objective

Truncated Gradient (Langford, Li & Zhang, 2009) approximately solves the $\ell_1$-penalized GLM

$$
\hat\beta \;=\; \arg\min_{\beta\in\mathbb{R}^p}\; \mathcal L(\beta) + \lambda\lVert\beta\rVert_1, \qquad
\nabla\ell_i(\beta) = -(y_i-\mu_i)\,x_i = (\mu(x_i^\top\beta)-y_i)\,x_i,
$$

by interleaving plain SGD steps with a **soft truncation** applied every $K$ steps. The
coordinatewise truncation operator, with gravity $\alpha=K\gamma\lambda$ and window $\theta$, is

$$
\boxed{\;
T(v,\alpha,\theta) =
\begin{cases}
\max(0,\, v-\alpha) & 0 \le v \le \theta,\\[2pt]
\min(0,\, v+\alpha) & -\theta \le v < 0,\\[2pt]
v & |v| > \theta,
\end{cases}
\;}
$$

i.e. shrink toward $0$ by $\alpha$ **but not past $0$**, and only **within the window**
$|v|\le\theta$ (coefficients larger than $\theta$ are left untouched).

## Algorithm

```text
Input: stream (x, y); step γ; λ; truncation period K; gravity g=γλ; window θ; init β⁽⁰⁾
1. β ← β⁽⁰⁾
2. for t = 1, 2, ... :
3.     receive (x, y)
4.     μ = g⁻¹(xᵀ β)                              # O(p)
5.     grad = (μ - y) · x                          # ∇ℓ_{i_t}(β), O(p)
6.     β ← β - γ · grad                            # plain SGD step, O(p)
7.     if t mod K == 0:                            # truncate periodically
8.         α = K · γ · λ
9.         for j = 1..p:  β_j ← T(β_j, α, θ)       # soft truncation, O(p)
Return β
```

- $K$ controls truncation frequency (amortizes the zeroing); the per-event gravity is
  $\alpha=K\gamma\lambda$ so the *average* shrinkage rate matches $\gamma\lambda$ per step.
- **Special case.** As $\theta\to\infty$ with $K=1$, $T(v,\gamma\lambda,\infty)=\mathcal S_{\gamma\lambda}(v)$
  is exact soft-thresholding, recovering **FOBOS-$\ell_1$**. Finite $\theta$ protects large
  coefficients from shrinkage (a SCAD/MCP-like effect).

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ | tuned | $\ell_1$ strength (gravity per step $=\gamma\lambda$) |
| $K$ | $1$–$10$ | truncation period; larger $K$ truncates less often, more strongly |
| $\theta$ | $\infty$ | truncation window; $\theta=\infty$ ⇒ soft-threshold; finite ⇒ protect large coefs |
| $\gamma$ | $\gamma_0 t^{-a}$ | SGD step size, $a\in(0.5,1]$ |
| init $\beta^{(0)}$ | $0$ | warm start permitted |

## Mapping to framework

- **Input:** stream of $(x_i,y_i)$, link $g$, $(\lambda, K, \theta)$, step $\gamma$.
- **Output:** sparse point estimate $\hat\beta$.
- **Links:** identity, logit, log.
- **Preprocessing:** feature scaling recommended so $\lambda$ acts uniformly across coordinates.

## Complexity

- **Per step:** one inner product and one SGD update, each $O(p)$; the periodic truncation is an
  $O(p)$ elementwise sweep (amortized $O(p/K)$ per step, but $O(p)$ worst case). Total $O(p)$ per
  observation.
- **Memory:** $O(p)$ — only the iterate $\beta$ — bounded and independent of the stream length;
  with sparse $x_{i_t}$ the gradient step is $O(\mathrm{nnz})$.

## Statistical guarantees

- **Online regret / sparsity trade-off.** Truncated Gradient enjoys an online regret bound and
  provably controls the induced sparsity through $(\lambda, K, \theta)$, with the truncation bias
  bounded by the gravity parameter [[Langford et al., 2009]](#ref-truncated_SGD_Langford_2009).
- It recovers FOBOS-$\ell_1$ (hence its regret guarantees) in the $\theta\to\infty$, $K=1$ limit.

## Variants & related

- [FOBOS](fobos.md) — special case ($\theta\to\infty$, $K=1$).
- [RDA](rda.md) — gradient-averaging alternative with stronger sparsity.
- [SGD](sgd.md) — the underlying unpenalized update.

## References

- <a id="ref-truncated_SGD_Langford_2009"></a> Langford, J., Li, L., and Zhang, T. (2009). Sparse online learning via truncated gradient. *J. Mach. Learn. Res.*, 10:777--801.
