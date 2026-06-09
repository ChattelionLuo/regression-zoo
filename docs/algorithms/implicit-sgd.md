---
id: implicit-sgd
name: "Implicit SGD (ISGD)"
aliases: ["proximal SGD", "implicit stochastic gradient descent", "normalized SGD"]
family: online-streaming
regime: [streaming, low-dim, high-dim]
penalty: none
link_support: [identity, logit, log]
output: point
year: 2017
refs: [Toulis2017, Fang2019]
status: draft
---

# Implicit SGD (ISGD)

!!! info "At a glance"
    **Family:** online-streaming · **Regime:** streaming / low-dim / high-dim · **Penalty:** none ·
    **Output:** point · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** [`Toulis and Airoldi, 2017`](#ref-Toulis2017) · [`Fang, 2019`](#ref-Fang2019)

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family)
  with canonical link; identity, logit, log are the primary cases.
- **Streaming / online**: points $(x_{i_t}, y_{i_t})$ arrive sequentially; storage is bounded to
  the running iterate.
- The per-sample gradient is **rank-1** in $x_i$:
  $\nabla\ell_i(\beta)=(\mu(x_i^\top\beta)-y_i)\,x_i$ — the property that makes the implicit
  update reduce to a scalar problem.

## Estimator / objective

Same target as SGD — the unpenalized GLM minimizer

$$
\widehat\beta \;=\; \arg\min_{\beta\in\mathbb{R}^p}\; \mathcal L(\beta), \qquad
\nabla\ell_i(\beta) = -(y_i-\mu_i)\,x_i = (\mu(x_i^\top\beta)-y_i)\,x_i,
$$

but with an **implicit** (backward-Euler / proximal) recursion in which the gradient is evaluated
at the **new** iterate:

$$
\boxed{\;
\beta^{(t+1)} \;=\; \beta^{(t)} - \gamma_t\,\nabla\ell_{i_t}\big(\beta^{(t+1)}\big).
\;}
$$

Equivalently $\beta^{(t+1)}=\operatorname{prox}_{\gamma_t \ell_{i_t}}(\beta^{(t)})
=\arg\min_\beta\{\ell_{i_t}(\beta)+\tfrac{1}{2\gamma_t}\lVert\beta-\beta^{(t)}\rVert_2^2\}$.

**1D reduction for GLMs.** Because $\nabla\ell_{i_t}$ points along $x_{i_t}$, the update keeps
$\beta^{(t+1)}=\beta^{(t)}+\xi_t\,x_{i_t}$ for a scalar $\xi_t$. Writing the implicit step as

$$
\beta^{(t+1)} \;=\; \beta^{(t)} + \gamma_t\big(y_{i_t}-\mu(x_{i_t}^\top\beta^{(t+1)})\big)\,x_{i_t},
$$

and taking the inner product with $x_{i_t}$ gives a **scalar fixed-point** in
$\eta^{+}=x_{i_t}^\top\beta^{(t+1)}$:

$$
\eta^{+} \;=\; x_{i_t}^\top\beta^{(t)} + \gamma_t\,\lVert x_{i_t}\rVert_2^2\,\big(y_{i_t}-\mu(\eta^{+})\big),
$$

solvable by 1D root-finding (the root is unique and bracketed since $\mu$ is monotone). The full
update is then $\beta^{(t+1)}=\beta^{(t)}+\gamma_t(y_{i_t}-\mu(\eta^{+}))\,x_{i_t}$.

## Algorithm

```text
Input: stream (x, y); step schedule γ_t; init β⁽⁰⁾
Optional: Polyak–Ruppert average β̄
1. β ← β⁽⁰⁾ ;  β̄ ← β⁽⁰⁾
2. for t = 0, 1, 2, ... :
3.     receive (x, y)
4.     η0 = xᵀ β ;  c = γ_t · ‖x‖²       # O(p)
5.     solve for η⁺:  η⁺ = η0 + c·(y - g⁻¹(η⁺))   # scalar root-find (monotone)
6.     ξ = γ_t · (y - g⁻¹(η⁺))            # scalar step magnitude
7.     β ← β + ξ · x                      # rank-1 update, O(p)
8.     β̄ ← β̄ + (β - β̄)/(t+1)             # optional averaging
Return β  (or averaged β̄ for inference)
```

The scalar solve in step 5 uses Newton/bisection on a monotone 1D equation and converges in a few
iterations; no $p\times p$ system is ever formed.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\gamma_0$ | problem-dependent | $\gamma_t=\gamma_0 t^{-a}$, $a\in(0.5,1]$; ISGD tolerates larger $\gamma_0$ than explicit SGD |
| $a$ (decay) | $0.5<a\le1$ | use $a\in(0.5,1)$ with averaging |
| 1D solver tol | $10^{-10}$ | tolerance for the scalar root $\eta^{+}$ |
| averaging | on (for inference) | Polyak–Ruppert $\bar\beta_t$ |
| init $\beta^{(0)}$ | $0$ | warm start permitted |
| standardization | recommended | shared learning rate across coordinates |

## Mapping to framework

- **Input:** stream of $(x_i,y_i)$, link $g$, schedule $\gamma_t$.
- **Output:** point estimate $\widehat\beta$, or averaged $\bar\beta_t$ with asymptotic covariance.
- **Links:** identity, logit, log (any monotone canonical $\mu=g^{-1}$).
- **Preprocessing:** feature scaling recommended; bounded memory, no data retention.

## Complexity

- **Per step:** one inner product ($O(p)$), a constant number of scalar function evaluations for
  the root (each $O(1)$ given $\eta_0$ and $\lVert x_{i_t}\rVert_2^2$), and one rank-1 update
  ($O(p)$). Total $O(p)$ time per observation.
- **Memory:** $O(p)$ — only the iterate (and $O(p)$ for the average); independent of the stream
  length.

## Statistical guarantees

- **Stability & consistency.** The implicit update is numerically stable — it cannot overshoot for
  large $\gamma_t$ the way explicit SGD can — while retaining the same limit $\beta^\star$ and the
  same asymptotic distribution as explicit SGD [`Toulis and Airoldi, 2017`](#ref-Toulis2017).
- **Asymptotic efficiency via averaging.** The Polyak–Ruppert average of implicit iterates is
  asymptotically normal with the MLE's sandwich covariance, enabling scalable inference
  [`Fang, 2019`](#ref-Fang2019).

## Variants & related

- [SGD](sgd.md) — explicit (forward) counterpart.
- [AdaGrad](adagrad.md) — adaptive step sizes (orthogonal idea, combinable).
- [FOBOS](fobos.md) — proximal step for an explicit *penalty* (vs. implicit *loss* here).
- [Renewable GLM](renewable-glm.md) — batch streaming with exact information accumulation.

## References

- <a id="ref-Toulis2017"></a> Toulis, P. and Airoldi, E. M. (2017). Asymptotic and finite-sample properties of estimators based on stochastic gradients. *Ann. Statist.*, 45(4):1694--1727.
- <a id="ref-Fang2019"></a> Fang, Y. (2019). Scalable statistical inference for averaged implicit stochastic gradient descent. *Scand. J. Stat.*, 46(4):987--1002.
