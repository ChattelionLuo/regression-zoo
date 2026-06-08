---
id: renewable-glm
name: "Renewable Estimation (streaming GLM)"
aliases: ["renewable estimation", "incremental GLM inference", "renewable GLM"]
family: online-streaming
regime: [streaming]
penalty: none
link_support: [any]
output: point+inference
year: 2020
refs: [luo2020renewable, Luo2020]
status: draft
---

# Renewable Estimation (streaming GLM)

!!! info "At a glance"
    **Family:** online-streaming · **Regime:** streaming · **Penalty:** none ·
    **Output:** point+inference · **Links:** any · **Status:** draft ·
    **Refs:** luo2020renewable, Luo2020

## Setting & assumptions

- Any GLM in the [exponential family](../framework/notation.md#2-exponential-dispersion-family)
  with any differentiable link $g$ (canonical or not).
- **Streaming in batches**: data arrive as a sequence of datasets
  $\mathcal D_1,\mathcal D_2,\dots$ (block $b$ has design $X_b\in\mathbb{R}^{n_b\times p}$,
  response $y_b$); raw batches are processed once and **discarded**.
- Bounded storage: only a $p$-vector estimate and a $p\times p$ aggregated information matrix are
  retained — never the accumulated data.

## Estimator / objective

The target is the **full-data MLE** that solves the aggregated score equation over all batches
seen through block $b$,

$$
\sum_{k=1}^{b} S_k(\beta) = 0, \qquad
S_k(\beta) = X_k^\top\big(y_k - \mu_k(\beta)\big), \quad \mu_k(\beta)=g^{-1}(X_k\beta),
$$

but computed **incrementally** (Luo & Song, 2020). Renewable estimation maintains a running
estimate $\hat\beta_{b}$ and the **aggregated negative Hessian / information**
$J_b=\sum_{k\le b} X_k^\top W_k X_k$. For a new batch $b$ it performs an information-weighted
one-step (Newton) update that fuses the previous summary with the current-batch score:

$$
\boxed{\;
\begin{aligned}
J_b &= J_{b-1} + X_b^\top W_b\big(\hat\beta_{b-1}\big)\,X_b,\\[4pt]
\hat\beta_b &= \hat\beta_{b-1} + J_b^{-1}\, X_b^\top\big(y_b - \mu_b(\hat\beta_{b-1})\big),
\end{aligned}
\;}
$$

where $W_b(\beta)=\operatorname{diag}\big(V(\mu_{b,i})\big)$ are the GLM weights (under the canonical
link). The term $J_{b-1}$ carries the **prior estimate weighted by accumulated information**, while
$X_b^\top(y_b-\mu_b(\hat\beta_{b-1}))$ is the new batch's score at the previous estimate — together
an aggregated estimating equation whose solution is the renewable estimator.

## Algorithm

```text
Input: stream of batches (X_b, y_b); link g; init β̂₀, J₀ = 0 (p×p)
1. for b = 1, 2, ... :
2.     receive batch (X_b, y_b)
3.     η = X_b · β̂_{b-1} ;  μ = g⁻¹(η)                  # O(n_b p)
4.     W = diag(V(μ))                                     # GLM weights
5.     score = X_bᵀ (y_b - μ)                            # current-batch score, O(n_b p)
6.     J_b = J_{b-1} + X_bᵀ W X_b                         # accumulate information, O(n_b p²)
7.     β̂_b = β̂_{b-1} + J_b⁻¹ · score                     # renewable (one-step) update, O(p³)
8.     discard (X_b, y_b) ; keep only β̂_b and J_b
Return β̂_b  and  Cov(β̂_b) ≈ J_b⁻¹
```

- A few inner Newton iterations within a batch (re-evaluating $\mu, W$, score at the updated
  estimate before committing $J_b$) sharpen the update when batches are large; one step suffices
  asymptotically.
- $J_b^{-1}$ doubles as the variance estimate for inference (see below).

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| link $g$ | any | canonical or non-canonical (use observed information if non-canonical) |
| inner Newton steps | 1 | per batch; more improves finite-batch accuracy |
| init $\hat\beta_0, J_0$ | $0,\,0$ | first batch behaves like a standard IRLS fit |
| batch size $n_b$ | data-driven | arbitrary, may vary across blocks |
| ridge on $J_b$ | optional | small $\epsilon I$ for early-batch invertibility |

## Mapping to framework

- **Input:** sequence of batches $(X_b, y_b)$, link $g$.
- **Output:** $\hat\beta_b$ **and** inference — covariance $\widehat{\operatorname{Cov}}(\hat\beta_b)=J_b^{-1}$
  (Wald intervals, tests). Hence `output: point+inference`.
- **Links:** any differentiable link.
- **Preprocessing:** consistent coding/scaling across batches; intercept via a ones column.

## Complexity

- **Per batch:** forming $X_b^\top W_b X_b$ and the score is $O(n_b p^2)$; the Newton solve is
  $O(p^3)$ (amortized small since $p$ is fixed and batches stream). Cost is **independent of the
  total number of batches** already processed.
- **Memory:** $O(p^2)$ — the estimate $\hat\beta_b$ ($O(p)$) plus the aggregated information $J_b$
  ($O(p^2)$). Crucially, **raw data are not stored**: storage is bounded by the summary, not the
  data volume.

## Statistical guarantees

- **Asymptotic equivalence to the full-data MLE.** As batches accumulate, $\hat\beta_b$ is
  consistent and asymptotically normal with the **same** limiting distribution as the oracle MLE
  computed on all data at once, with $J_b^{-1}$ a consistent variance estimator
  [[Luo and Song, 2020]](#ref-luo2020renewable).
- This yields valid streaming confidence intervals and tests while storing only $O(p^2)$ summary
  statistics.

## Variants & related

- [SGD](sgd.md) · [Implicit SGD](implicit-sgd.md) — first-order streaming alternatives ($O(p)$ memory, no information matrix).
- [GLM via IRLS](glm-irls.md) — the batch (full-data) Newton/Fisher-scoring solver that renewable estimation tracks incrementally.
- [OLS](ols.md) — Gaussian/identity special case; the renewable update reduces to recursive least squares.

## References

- <a id="ref-luo2020renewable"></a><a id="ref-Luo2020"></a> Luo, L. and Song, P. X.-K. (2020). Renewable estimation and incremental inference in generalized linear models with streaming data sets. *J. R. Stat. Soc. Ser. B*, 82(1):69--97.
