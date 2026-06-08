---
id: fobos
name: "FOBOS (Forward-Backward Splitting)"
aliases: ["forward-backward splitting", "proximal SGD with regularization", "FOLOS"]
family: online-streaming
regime: [streaming, high-dim, low-dim]
penalty: lasso
link_support: [identity, logit, log]
output: point
year: 2009
refs: [xiao2009dual, truncated_SGD_Langford_2009]
status: draft
---

# FOBOS (Forward-Backward Splitting)

!!! info "At a glance"
    **Family:** online-streaming · **Regime:** streaming / high-dim / low-dim · **Penalty:** lasso (any prox-able) ·
    **Output:** point · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** xiao2009dual, truncated_SGD_Langford_2009

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family)
  with canonical link; identity, logit, log are the primary cases.
- **Streaming / online**: points arrive sequentially; per-step cost and storage are bounded.
- A convex, prox-able penalty $P$ (canonically $\ell_1$ for online sparsity); high-dimensional
  sparse $\beta^\star$ is the motivating regime.

## Estimator / objective

FOBOS (Duchi & Singer, 2009) minimizes the **composite** online objective

$$
\hat\beta \;=\; \arg\min_{\beta\in\mathbb{R}^p}\; \mathcal L(\beta) + \lambda P(\beta), \qquad
\nabla\ell_i(\beta) = -(y_i-\mu_i)\,x_i = (\mu(x_i^\top\beta)-y_i)\,x_i,
$$

by **splitting** each iteration into an explicit (forward) subgradient step on the loss and an
implicit (backward) proximal step on the penalty:

$$
\boxed{\;
\begin{aligned}
\text{(i) forward:}\quad & \beta^{(t+1/2)} = \beta^{(t)} - \gamma_t\,\nabla\ell_{i_t}(\beta^{(t)})
   = \beta^{(t)} - \gamma_t\big(\mu(x_{i_t}^\top\beta^{(t)})-y_{i_t}\big)\,x_{i_t},\\[2pt]
\text{(ii) backward:}\quad & \beta^{(t+1)} = \arg\min_{\beta}\;\tfrac12\lVert\beta-\beta^{(t+1/2)}\rVert_2^2
   + \gamma_t\lambda\,P(\beta) \;=\; \operatorname{prox}_{\gamma_t\lambda P}\big(\beta^{(t+1/2)}\big).
\end{aligned}
\;}
$$

**$\ell_1$ case.** The backward step is coordinatewise **soft-thresholding**, producing exact
zeros (sparse online iterates):

$$
\beta^{(t+1)}_j = \mathcal S_{\gamma_t\lambda}\big(\beta^{(t+1/2)}_j\big)
   = \operatorname{sign}(\beta^{(t+1/2)}_j)\,\big(|\beta^{(t+1/2)}_j|-\gamma_t\lambda\big)_+.
$$

## Algorithm

```text
Input: stream (x, y); step schedule γ_t; penalty λP (prox known); init β⁽⁰⁾
1. β ← β⁽⁰⁾
2. for t = 0, 1, 2, ... :
3.     receive (x, y)
4.     μ = g⁻¹(xᵀ β)                          # O(p)
5.     g = (μ - y) · x                         # forward gradient, O(p)
6.     v = β - γ_t · g                         # forward (half) step, O(p)
7.     β ← prox_{γ_t·λ·P}(v)                   # backward step
       # for P = ‖·‖₁:  β_j = Soft(v_j, γ_t·λ)  (soft-threshold), O(p)
Return β
```

- Step sizes follow $\gamma_t=\gamma_0 t^{-a}$, $a\in(0.5,1]$ (Robbins–Monro conditions).
- Any prox-able penalty fits: $\ell_1$ (soft-threshold), $\ell_2^2$/ridge (shrinkage),
  group lasso (block soft-threshold), elastic net.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ | tuned | regularization strength; controls online sparsity |
| $P$ | $\ell_1$ | any prox-able penalty |
| $\gamma_0$ | problem-dependent | base learning rate |
| $a$ (decay) | $0.5<a\le1$ | $\gamma_t=\gamma_0 t^{-a}$ |
| averaging | optional | tail-average iterates to stabilize |
| init $\beta^{(0)}$ | $0$ | warm start permitted |

## Mapping to framework

- **Input:** stream of $(x_i,y_i)$, link $g$, penalty $(\lambda, P)$, schedule $\gamma_t$.
- **Output:** sparse point estimate $\hat\beta$.
- **Links:** identity, logit, log.
- **Preprocessing:** feature scaling recommended so a single $\lambda$ acts uniformly.

## Complexity

- **Per step:** one inner product ($O(p)$), one axpy forward step ($O(p)$), and one elementwise
  prox/soft-threshold ($O(p)$). Total $O(p)$ time per observation.
- **Memory:** $O(p)$ — only the iterate $\beta$ — bounded and independent of the stream length.
  With sparse $x_{i_t}$, the forward step touches only $O(\mathrm{nnz})$ coordinates.

## Statistical guarantees

- **Online regret.** For convex losses with step sizes $\gamma_t\propto 1/\sqrt{t}$, FOBOS attains
  $\mathcal O(\sqrt{T})$ regret against the best fixed $\beta$ for the composite objective
  (Duchi & Singer, 2009).
- **Sparsity behavior.** Single-point soft-thresholding zeros only coordinates whose half-step
  magnitude falls below $\gamma_t\lambda$; because $\gamma_t\to0$, FOBOS yields **weaker** online
  sparsity than gradient-averaging schemes such as [RDA](rda.md) [[Xiao, 2010]](#ref-xiao2009dual).

## Variants & related

- [RDA](rda.md) — averages gradients before thresholding, giving stronger/more stable sparsity.
- [Truncated Gradient](truncated-gradient.md) — recovers FOBOS-$\ell_1$ as a special case.
- [SGD](sgd.md) — the forward step alone; [AdaGrad](adagrad.md) — adaptive composite variant.

## References

- <a id="ref-duchi2009fobos"></a> Duchi, J. and Singer, Y. (2009). Efficient online and batch learning using forward-backward splitting. *J. Mach. Learn. Res.*, 10:2899--2934.
- <a id="ref-xiao2009dual"></a> Xiao, L. (2010). Dual averaging methods for regularized stochastic learning and online optimization. *J. Mach. Learn. Res.*, 11:2543--2596.
- <a id="ref-truncated_SGD_Langford_2009"></a> Langford, J., Li, L., and Zhang, T. (2009). Sparse online learning via truncated gradient. *J. Mach. Learn. Res.*, 10:777--801.
