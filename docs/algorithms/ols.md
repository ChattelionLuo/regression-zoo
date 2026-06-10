---
id: ols
name: "Ordinary Least Squares (OLS)"
aliases: ["linear regression", "normal equations", "Gaussian MLE (identity link)"]
family: classical
regime: [low-dim]
penalty: none
link_support: [identity]
output: point
year: 1805
refs: [Nocedal2006]
status: reviewed
---

# Ordinary Least Squares (OLS)

!!! info "At a glance"
    **Family:** classical · **Regime:** low-dim ($n>p$) · **Penalty:** none ·
    **Output:** point · **Links:** identity · **Status:** reviewed

## Setting & assumptions

- Gaussian family with **identity link**: $\;y = X\beta^\star + \varepsilon$, $\mathbb{E}[\varepsilon\mid X]=0$.
- Low-dimensional: $n \ge p$ and $X$ has **full column rank** ($X^\top X \succ 0$).
- Homoskedastic errors $\operatorname{Var}(\varepsilon)=\sigma^2 I_n$ for the classical
  Gauss–Markov optimality statement (not needed for the estimator itself).

## Estimator / objective

OLS is the unpenalized template ($P\equiv 0$) with the Gaussian loss:

$$
\widehat\beta \;=\; \arg\min_{\beta\in\mathbb{R}^p}\; \frac{1}{2n}\,\lVert y - X\beta\rVert_2^2 .
$$

Setting the gradient $-\tfrac1n X^\top(y-X\beta)$ to zero gives the **normal equations**
$X^\top X\,\beta = X^\top y$, whose unique solution (full rank) is

$$
\boxed{\;\widehat\beta = (X^\top X)^{-1} X^\top y\;}
$$

## Algorithm

Closed form; computed stably via a matrix factorization rather than inverting $X^\top X$.

```text
Input: X (n×p, full rank), y (n)
1. Compute thin QR:  X = Q R          # Q: n×p orthonormal, R: p×p upper-triangular
2. Solve R β = Qᵀ y  by back-substitution
Return β̂
```

Equivalent routes: Cholesky of $X^\top X$, or SVD $X=U\Sigma V^\top$ giving
$\widehat\beta = V\Sigma^{-1}U^\top y$ (most robust when $X$ is ill-conditioned).

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| intercept | included | prepend a column of ones, or center $X,y$ first |
| solver | QR | QR / Cholesky / SVD; SVD safest for ill-conditioning |
| weights $w_i$ | none | if supplied → **WLS**: minimize $\sum w_i (y_i-x_i^\top\beta)^2$ |

No regularization parameter ($\lambda=0$).

## Mapping to framework

- **Input:** $X\in\mathbb{R}^{n\times p}$, $y\in\mathbb{R}^n$, link `identity`.
- **Output:** $\widehat\beta=(X^\top X)^{-1}X^\top y$.
- **Links:** identity only (for other links use [GLM-IRLS](glm-irls.md)).
- **Preprocessing:** center/scale optional; affects intercept handling only.

## Complexity

- Time: $O(np^2)$ to form/factor, $O(p^3)$ for the triangular solve. Memory $O(np)$.

## Statistical guarantees

- Unbiased: $\mathbb{E}[\widehat\beta]=\beta^\star$; covariance $\sigma^2(X^\top X)^{-1}$.
- **Gauss–Markov:** BLUE among linear unbiased estimators under homoskedasticity.
- Under Gaussian errors, $\widehat\beta$ is the MLE and is efficient.

## Variants & related

- [Ridge Regression](ridge.md) — add $\lambda\lVert\beta\rVert_2^2$ for ill-conditioned/high-dim $X$.
- [GLM via IRLS](glm-irls.md) — non-identity links / non-Gaussian families.
- **WLS / GLS** — weighted / generalized least squares for heteroskedastic/correlated errors.

## References

- <a id="ref-Nocedal2006"></a> <a href="https://link.springer.com/book/10.1007/978-0-387-40065-5" class="ref-link" target="_blank" rel="noopener noreferrer">Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer.</a>
