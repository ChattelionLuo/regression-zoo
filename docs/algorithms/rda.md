---
id: rda
name: "Regularized Dual Averaging (RDA)"
aliases: ["dual averaging", "RDA", "regularized dual averaging method"]
family: online-streaming
regime: [streaming, high-dim, low-dim]
penalty: lasso
link_support: [identity, logit, log]
output: point
year: 2009
refs: [xiao2009dual]
status: draft
---

# Regularized Dual Averaging (RDA)

!!! info "At a glance"
    **Family:** online-streaming · **Regime:** streaming / high-dim / low-dim · **Penalty:** lasso ·
    **Output:** point · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** [`Xiao, 2010`](#ref-xiao2009dual)

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family)
  with canonical link; identity, logit, log are the primary cases.
- **Streaming / online**: points arrive sequentially; per-step cost and storage are bounded.
- A convex regularizer $P$ (canonically $\ell_1$); sparse $\beta^\star$ in high dimension is the
  motivating regime.

## Estimator / objective

RDA minimizes the composite online objective

$$
\widehat\beta \;=\; \arg\min_{\beta\in\mathbb{R}^p}\; \mathcal L(\beta) + \lambda P(\beta), \qquad
g_t := \nabla\ell_{i_t}(\beta^{(t)}) = \big(\mu(x_{i_t}^\top\beta^{(t)})-y_{i_t}\big)\,x_{i_t},
$$

but, unlike FOBOS, it forms the **running average of all past subgradients** and minimizes against
it plus the penalty and a vanishing proximal term:

$$
\bar g_t \;=\; \frac1t\sum_{s=1}^{t} g_s,
\qquad
\boxed{\;
\beta^{(t+1)} = \arg\min_{\beta\in\mathbb{R}^p}\;\Big\{\, \bar g_t^\top\beta \;+\; \lambda P(\beta)
   \;+\; \frac{\beta_t}{t}\, h(\beta) \,\Big\},
\;}
$$

where $h$ is a strongly convex auxiliary (canonically $h(\beta)=\tfrac12\lVert\beta\rVert_2^2$) and
$\{\beta_t\}$ is a nondecreasing scale (e.g. $\beta_t=\gamma\sqrt{t}$).

**$\ell_1$ closed form.** With $h(\beta)=\tfrac12\lVert\beta\rVert_2^2$ and $P=\lVert\cdot\rVert_1$,
the update is a coordinatewise **soft-threshold on the averaged gradient**:

$$
\beta^{(t+1)}_j = -\frac{t}{\beta_t}\,\operatorname{sign}(\bar g_{t,j})\,\big(|\bar g_{t,j}|-\lambda\big)_+
= -\frac{t}{\beta_t}\,\mathcal S_{\lambda}(\bar g_{t,j}),
$$

so $\beta^{(t+1)}_j=0$ **whenever** $|\bar g_{t,j}|\le\lambda$ — a threshold that does not shrink
with $t$, giving stronger and more stable sparsity than FOBOS.

## Algorithm

```text
Input: stream (x, y); scale schedule β_t (e.g. γ√t); λ; aux h; init β⁽⁰⁾=0
1. ḡ ← 0  (length-p running average of gradients)
2. for t = 1, 2, ... :
3.     receive (x, y)
4.     μ = g⁻¹(xᵀ β)                          # O(p)
5.     g = (μ - y) · x                         # subgradient g_t, O(p)
6.     ḡ ← ḡ + (g - ḡ)/t                       # update running average, O(p)
7.     # closed form for h = ½‖·‖², P = ‖·‖₁:
8.     for j = 1..p:  β_j = -(t/β_t) · Soft(ḡ_j, λ)     # O(p)
Return β
```

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ | tuned | $\ell_1$ strength; threshold on $\bar g_t$ |
| $\beta_t$ | $\gamma\sqrt{t}$ | nondecreasing proximal scale; $\gamma>0$ tunes aggressiveness |
| $h$ | $\tfrac12\lVert\cdot\rVert_2^2$ | strongly convex auxiliary (enables closed form) |
| $P$ | $\ell_1$ | other convex regularizers admissible |
| init $\beta^{(0)}$ | $0$ | $\bar g_0=0$ |

## Mapping to framework

- **Input:** stream of $(x_i,y_i)$, link $g$, penalty $(\lambda, P)$, schedule $\beta_t$, auxiliary $h$.
- **Output:** sparse point estimate $\widehat\beta$.
- **Links:** identity, logit, log.
- **Preprocessing:** feature scaling recommended so $\lambda$ thresholds coordinates uniformly.

## Complexity

- **Per step:** one inner product ($O(p)$), a running-average update ($O(p)$), and a closed-form
  coordinatewise threshold ($O(p)$). Total $O(p)$ time per observation.
- **Memory:** $O(p)$ — the iterate $\beta$ and the averaged gradient $\bar g_t$ — bounded and
  independent of the stream length.

## Statistical guarantees

- **Online regret.** With $\beta_t\propto\sqrt{t}$, RDA attains $\mathcal O(\sqrt{T})$ regret for
  convex losses [`Xiao, 2010`](#ref-xiao2009dual).
- **Sparsity.** Because the threshold acts on the *averaged* gradient $\bar g_t$ with a level that
  does not vanish, RDA produces sparser, more stable solutions than the diminishing-threshold
  FOBOS, as analyzed in [`Xiao, 2010`](#ref-xiao2009dual).

## Variants & related

- [FOBOS](fobos.md) — thresholds the current half-step (weaker sparsity).
- [Truncated Gradient](truncated-gradient.md) — alternative online-$\ell_1$ truncation scheme.
- [AdaGrad](adagrad.md) — adaptive (per-coordinate) dual-averaging form.

## References

- <a id="ref-xiao2009dual"></a> Xiao, L. (2010). Dual averaging methods for regularized stochastic learning and online optimization. *J. Mach. Learn. Res.*, 11:2543--2596.
