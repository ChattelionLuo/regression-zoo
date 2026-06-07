---
id: sgd
name: "Stochastic Gradient Descent (SGD)"
aliases: ["online gradient descent", "Robbins-Monro", "stochastic approximation"]
family: online-streaming
regime: [streaming, low-dim, high-dim]
penalty: none
link_support: [identity, logit, log]
output: point
year: 1951
refs: [Robbins1951, polyak1992acceleration, ruppert1988efficient, chen2020statistical]
status: draft
---

# Stochastic Gradient Descent (SGD)

!!! info "At a glance"
    **Family:** online-streaming · **Regime:** streaming / low-dim / high-dim · **Penalty:** none ·
    **Output:** point · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** Robbins1951, polyak1992acceleration, ruppert1988efficient, chen2020statistical

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family)
  with canonical link; identity (Gaussian), logit (Bernoulli), log (Poisson) are the primary cases.
- **Streaming / online**: observations $(x_{i_t}, y_{i_t})$ arrive one at a time (or are sampled
  i.i.d. from a fixed set); only the current point and the running iterate are held in memory.
- The population risk $\mathbb{E}[\ell(\beta)]$ is convex (strictly so under a full-rank design)
  with a unique minimizer $\beta^\star$; gradients are unbiased estimates of $\nabla\mathcal L$.

## Estimator / objective

SGD targets the unpenalized GLM minimizer ($P\equiv 0$):

$$
\hat\beta \;=\; \arg\min_{\beta\in\mathbb{R}^p}\; \mathcal L(\beta)
            \;=\; \arg\min_{\beta\in\mathbb{R}^p}\; \frac1n\sum_{i=1}^n \ell_i(\beta),
$$

using the **per-sample gradient** of the GLM mean negative log-likelihood,

$$
\nabla \ell_i(\beta) \;=\; -\big(y_i - \mu_i\big)\, x_i
                     \;=\; \big(\mu(x_i^\top\beta) - y_i\big)\, x_i,
\qquad \mu_i = g^{-1}(x_i^\top\beta).
$$

The defining recursion (Robbins–Monro stochastic approximation) is

$$
\boxed{\;
\beta^{(t+1)} \;=\; \beta^{(t)} - \gamma_t\,\nabla \ell_{i_t}(\beta^{(t)})
              \;=\; \beta^{(t)} - \gamma_t\big(\mu(x_{i_t}^\top\beta^{(t)}) - y_{i_t}\big)\,x_{i_t}.
\;}
$$

## Algorithm

```text
Input: stream (x_{i_t}, y_{i_t}); step schedule γ_t; init β⁽⁰⁾ (e.g. 0)
Optional: Polyak–Ruppert average β̄
1. β ← β⁽⁰⁾ ;  β̄ ← β⁽⁰⁾
2. for t = 0, 1, 2, ... :
3.     receive (x, y)
4.     η = xᵀ β                         # linear predictor, O(p)
5.     μ = g⁻¹(η)                        # inverse link
6.     g_t = (μ - y) · x                 # ∇ℓ_{i_t}(β), rank-1 in x
7.     β ← β - γ_t · g_t                 # SGD step, O(p)
8.     β̄ ← β̄ + (β - β̄)/(t+1)            # running average (optional)
Return β  (or the averaged β̄ for inference)
```

- **Step-size schedules.** A decaying rate $\gamma_t = \gamma_0\, t^{-a}$ with $a\in(0.5, 1]$
  satisfies the Robbins–Monro conditions $\sum_t\gamma_t=\infty$, $\sum_t\gamma_t^2<\infty$ and
  guarantees a.s. convergence. $a=1$ is the classical choice; $a\in(0.5,1)$ is preferred when it
  is combined with averaging.
- **Polyak–Ruppert averaging.** Report the tail average
  $\bar\beta_t = \frac1t\sum_{s=1}^{t}\beta^{(s)}$ rather than the last iterate. With a slowly
  decaying $\gamma_t$ ($a<1$), $\bar\beta_t$ is **asymptotically efficient** — it attains the same
  limiting variance as the MLE despite using only first-order updates.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\gamma_0$ | problem-dependent | base learning rate; too large diverges, too small crawls |
| $a$ (decay) | $0.5 < a \le 1$ | $\gamma_t=\gamma_0 t^{-a}$; use $a\in(0.5,1)$ with averaging |
| averaging | on (for inference) | Polyak–Ruppert tail/running average $\bar\beta_t$ |
| init $\beta^{(0)}$ | $0$ | warm start permitted |
| passes / epochs | 1 (true streaming) | multiple passes if data are revisitable |
| standardization | recommended | scale columns so a single $\gamma_t$ suits all coordinates |

## Mapping to framework

- **Input:** stream of $(x_i, y_i)$, link $g$, schedule $\gamma_t$.
- **Output:** point estimate $\hat\beta$ (last iterate), or $\bar\beta_t$ with an asymptotic
  covariance estimate (`point+inference` via averaging).
- **Links:** identity, logit, log (any canonical link supplying $\mu=g^{-1}(\eta)$).
- **Preprocessing:** feature scaling strongly recommended; no full-data storage required.

## Complexity

- **Per step:** one inner product $x_{i_t}^\top\beta$ and one rank-1 axpy update, both $O(p)$;
  no matrix is formed. Total $O(p)$ time per observation.
- **Memory:** $O(p)$ — the iterate $\beta$ (plus $O(p)$ for the running average). Bounded and
  independent of the number of observations seen, which is what makes SGD suitable for streams.

## Statistical guarantees

- **Consistency.** Under the Robbins–Monro step conditions, $\beta^{(t)}\to\beta^\star$ a.s.
  (`Robbins1951`).
- **Asymptotic normality of the averaged iterate.** With $\gamma_t=\gamma_0 t^{-a}$, $a\in(0.5,1)$,
  $\sqrt{t}\,(\bar\beta_t - \beta^\star)\xrightarrow{d}\mathcal N(0,\, H^{-1} G H^{-1})$ where
  $H=\nabla^2\mathcal L(\beta^\star)$ and $G$ is the gradient-noise covariance; this matches the
  MLE's efficiency (`polyak1992acceleration`, `ruppert1988efficient`).
- **Inference.** Plug-in / online estimators of the sandwich covariance permit confidence
  intervals from the averaged SGD path (`chen2020statistical`).

## Variants & related

- [Implicit SGD](implicit-sgd.md) — proximal/implicit update for stability.
- [AdaGrad](adagrad.md) — per-coordinate adaptive step sizes.
- [FOBOS](fobos.md) · [RDA](rda.md) · [Truncated Gradient](truncated-gradient.md) — regularized/sparse online variants.
- [Renewable GLM](renewable-glm.md) — batch streaming with full-data efficiency.

## References

- Robbins & Monro (1951), *A stochastic approximation method* (`Robbins1951`).
- Polyak & Juditsky (1992), *Acceleration of stochastic approximation by averaging* (`polyak1992acceleration`).
- Ruppert (1988), *Efficient estimations from a slowly convergent Robbins–Monro process* (`ruppert1988efficient`).
- Chen, Lee, Tong & Zhang (2020), *Statistical inference for model parameters in stochastic gradient descent* (`chen2020statistical`).
