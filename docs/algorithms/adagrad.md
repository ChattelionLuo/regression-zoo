---
id: adagrad
name: "AdaGrad"
aliases: ["adaptive subgradient method", "adaptive gradient"]
family: online-streaming
regime: [streaming, low-dim, high-dim]
penalty: none
link_support: [identity, logit, log]
output: point
year: 2011
refs: [AdaGrad2011]
status: draft
---

# AdaGrad

!!! info "At a glance"
    **Family:** online-streaming · **Regime:** streaming / low-dim / high-dim · **Penalty:** none (or composite) ·
    **Output:** point · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** [`Duchi et al., 2011`](#ref-AdaGrad2011)

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family)
  with canonical link; identity, logit, log are the primary cases.
- **Streaming / online**: points arrive sequentially; per-step compute and storage are bounded.
- Especially effective with **sparse / heterogeneously scaled features**, where a single global
  learning rate is poorly suited and per-coordinate adaptation helps.

## Estimator / objective

AdaGrad targets the unpenalized GLM minimizer (a composite penalty $P$ is optional, see below):

$$
\widehat\beta \;=\; \arg\min_{\beta\in\mathbb{R}^p}\; \mathcal L(\beta), \qquad
g_t := \nabla\ell_{i_t}(\beta^{(t)}) = \big(\mu(x_{i_t}^\top\beta^{(t)})-y_{i_t}\big)\,x_{i_t}.
$$

It accumulates the **per-coordinate** sum of squared gradients and rescales each step by it. With
$G_t = G_{t-1} + g_t\odot g_t$ (elementwise, $G_0=0$), the **diagonal** update is

$$
\boxed{\;
\beta^{(t+1)}_j \;=\; \beta^{(t)}_j \;-\; \frac{\eta}{\sqrt{G_{t,j}}+\epsilon}\; g_{t,j},
\qquad G_{t,j} = \sum_{s=1}^{t} g_{s,j}^2 .
\;}
$$

Equivalently $\beta^{(t+1)}=\beta^{(t)}-\eta\,(\,\sqrt{G_t}+\epsilon\,)^{-1}\!\odot g_t$, the
diagonal restriction of the full-matrix update $\beta^{(t+1)}=\beta^{(t)}-\eta\,H_t^{-1}g_t$ with
$H_t=(\sum_{s\le t} g_s g_s^\top)^{1/2}$ (the diagonal form is the one used in practice).

## Algorithm

```text
Input: stream (x, y); base rate η; epsilon ε; init β⁽⁰⁾
1. β ← β⁽⁰⁾ ;  G ← 0  (length-p accumulator)
2. for t = 1, 2, ... :
3.     receive (x, y)
4.     η_lin = xᵀ β ;  μ = g⁻¹(η_lin)           # O(p)
5.     g = (μ - y) · x                           # ∇ℓ_{i_t}(β), O(p)
6.     G ← G + g ⊙ g                             # accumulate, O(p)
7.     for j = 1..p:  β_j ← β_j - η · g_j / (sqrt(G_j) + ε)   # O(p)
Return β
```

**Composite / proximal variant.** For a regularizer $\lambda P$ (e.g. $\ell_1$), replace step 7 by
an adaptive proximal step,

$$
\beta^{(t+1)} = \operatorname*{arg\,min}_{\beta}\;\Big\{ g_t^\top\beta + \lambda P(\beta)
   + \tfrac{1}{2\eta}\,(\beta-\beta^{(t)})^\top \big(\sqrt{G_t}+\epsilon\big)\,(\beta-\beta^{(t)}) \Big\},
$$

which for $P=\lVert\cdot\rVert_1$ is a per-coordinate soft-threshold with coordinate-specific
step $\eta/(\sqrt{G_{t,j}}+\epsilon)$, giving sparse adaptive online solutions.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\eta$ (base rate) | $\sim 10^{-1}$ | a single global scale; adaptation handles per-coordinate tuning |
| $\epsilon$ | $10^{-8}$ | numerical floor preventing division by zero |
| $G_0$ | $0$ | accumulator initialization (sometimes a small positive offset) |
| $\lambda, P$ | none | optional composite/prox regularizer |
| init $\beta^{(0)}$ | $0$ | warm start permitted |

## Mapping to framework

- **Input:** stream of $(x_i,y_i)$, link $g$, base rate $\eta$ (optional $\lambda,P$).
- **Output:** point estimate $\widehat\beta$ (with sparsity if a prox penalty is used).
- **Links:** identity, logit, log.
- **Preprocessing:** AdaGrad is comparatively robust to feature scaling, but centering still helps.

## Complexity

- **Per step:** one inner product, one elementwise square-accumulate, and one elementwise scaled
  update — each $O(p)$. Total $O(p)$ time per observation; no $p\times p$ matrix (the full-matrix
  form would be $O(p^2)$–$O(p^3)$ and is avoided).
- **Memory:** $O(p)$ for $\beta$ plus $O(p)$ for the accumulator $G$ — bounded and independent of
  the number of observations.

## Statistical guarantees

- **Online regret.** AdaGrad achieves data-dependent regret
  $\mathcal O\!\big(\max_j \lVert g_{1:T,j}\rVert_2\big)$, never worse than and often far better
  than non-adaptive online gradient descent, with the largest gains on sparse/predictable
  features [`Duchi et al., 2011`](#ref-AdaGrad2011).
- It is an optimization (regret) guarantee rather than an asymptotic-efficiency statement; for
  parameter inference, pair with averaging as in [SGD](sgd.md).

## Variants & related

- [SGD](sgd.md) — non-adaptive baseline.
- [Implicit SGD](implicit-sgd.md) — stability via implicit updates.
- [FOBOS](fobos.md) · [RDA](rda.md) — the composite/prox variant connects AdaGrad to these regularized methods.

## References

- <a id="ref-AdaGrad2011"></a> Duchi, J., Hazan, E., and Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. *J. Mach. Learn. Res.*, 12:2121--2159.
