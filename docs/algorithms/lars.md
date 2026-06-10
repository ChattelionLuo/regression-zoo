---
id: lars
name: "Least Angle Regression (LARS)"
aliases: ["LARS", "least angle regression", "LARS-Lasso", "homotopy / lasso path"]
family: path-homotopy
regime: [low-dim, high-dim]
penalty: lasso
link_support: [identity]
output: path
year: 2004
refs: [tibshirani1996regression, buhlmann2011statistics]
status: draft
---

# Least Angle Regression (LARS)

!!! info "At a glance"
    **Family:** path-homotopy · **Regime:** low-dim / high-dim · **Penalty:** lasso ($\ell_1$) ·
    **Output:** path · **Links:** identity · **Status:** draft ·
    **Refs:** [`Tibshirani, 1996`](#ref-tibshirani1996regression) · [`Buhlmann and van de Geer, 2011`](#ref-buhlmann2011statistics)

## Setting & assumptions

- Gaussian family with **identity link**: $\;y = X\beta^\star + \varepsilon$.
- Columns of $X$ **standardized** to mean $0$ and unit $\ell_2$-norm, and $y$ centered, so that
  correlations $X_{\cdot j}^\top r$ are comparable across coordinates and the intercept is handled
  separately.
- Works in low- and high-dimensional ($p\gtrsim n$) regimes; the motivating output is the **entire
  solution path**, not a single fit.
- The plain LARS step assumes correlations can be kept tied; the **LARS-Lasso** modification adds a
  sign-agreement rule so that the path coincides exactly with the lasso path.

## Estimator / objective

The LARS-Lasso modification traces the full $\ell_1$-regularization path, i.e. the family of
minimizers of the lasso objective over all $\lambda\ge 0$:

$$
\widehat\beta(\lambda) \;=\; \arg\min_{\beta\in\mathbb{R}^p}\;
\frac{1}{2}\,\lVert y - X\beta\rVert_2^2 \;+\; \lambda\,\lVert\beta\rVert_1 .
$$

Because this solution is **piecewise linear in $\lambda$**, it is fully described by its breakpoints
(knots). LARS computes those knots and the linear segments between them directly, rather than
solving the optimization on a $\lambda$-grid. Plain (unmodified) LARS solves a slightly less
constrained problem; adding the drop/sign rule below makes it reproduce $\widehat\beta(\lambda)$ exactly.

## Algorithm

LARS maintains an **active set** $\mathcal A$ of variables that are tied for maximal absolute
correlation with the current residual, and moves the coefficients in the **equiangular direction**
— the direction making equal angles with all active predictors — so that those correlations stay
tied and decrease together until a new variable catches up.

Let $r = y - X\widehat\beta$ be the residual, $c = X^\top r$ the current correlations, and
$C = \max_j |c_j|$. With $\mathcal A$ the active set, signs $s_j=\operatorname{sign}(c_j)$, and
$X_{\mathcal A}$ the signed active columns, define

$$
G = X_{\mathcal A}^\top X_{\mathcal A}, \quad
A = \big(\mathbf 1^\top G^{-1}\mathbf 1\big)^{-1/2}, \quad
w = A\,G^{-1}\mathbf 1, \quad
u = X_{\mathcal A}\,w ,
$$

where $u$ is the unit **equiangular vector** ($X_{\mathcal A}^\top u = A\mathbf 1$, $\lVert u\rVert_2=1$).
The step length $\gamma$ to the next join is the **min-ratio rule**

$$
\gamma \;=\; \min_{j\notin\mathcal A}^{+}
\left\{ \frac{C - c_j}{A - a_j},\; \frac{C + c_j}{A + a_j} \right\},
\qquad a = X^\top u ,
$$

($\min^{+}$ = minimum over positive entries). For **LARS-Lasso**, also compute the distance at which
an active coefficient would hit zero,
$\gamma_{\text{drop}} = \min^{+}_{j\in\mathcal A}\{-\widehat\beta_j / (s_j w_j)\}$; if
$\gamma_{\text{drop}} < \gamma$, take that shorter step and **drop** the offending variable (its
sign no longer agrees), which is exactly what enforces lasso sign agreement.

```text
Input: X (standardized), y (centered)
Init:  β = 0;  r = y;  A = ∅
repeat:
    c = Xᵀ r;  C = max_j |c_j|
    A = { j : |c_j| = C }            # variables tied at max correlation
    s_j = sign(c_j) for j in A
    G = X_Aᵀ X_A
    A_norm = (1ᵀ G⁻¹ 1)^(-1/2)
    w = A_norm · G⁻¹ 1               # coefficients of equiangular direction
    u = X_A w                        # unit equiangular vector
    a = Xᵀ u
    γ = min⁺_{j∉A} { (C - c_j)/(A_norm - a_j),  (C + c_j)/(A_norm + a_j) }
    # --- LARS-Lasso modification ---
    γ_drop = min⁺_{j∈A} { -β_j / (s_j w_j) }
    if γ_drop < γ:
        γ = γ_drop;  β += γ · (signed w into full vector);  drop that j from A
    else:
        β += γ · (signed w into full vector)   # a new variable joins next round
    r = y - X β
until correlations ≈ 0  (or all variables active / desired λ reached)
Return path {β at each knot}
```

Each iteration changes the active set by one variable, so the **full piecewise-linear path** is
obtained in roughly $O(p)$ steps (in the low-dimensional, no-drop case the path has $\approx p$
knots).

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| mode | `lasso` | `lasso` (with drop/sign rule) reproduces the lasso path; plain `lar` omits it |
| standardize | true | columns to unit $\ell_2$-norm; intercept from centering |
| stopping | full path | stop at a target #steps, target $\lambda$, or when $\mathcal A$ saturates |
| tie/precision tol | $10^{-12}$ | guards the equiangular solve and min-ratio comparisons |

There is no $\lambda$ to tune *a priori* — LARS returns the whole path; a single $\widehat\beta(\lambda)$
is read off (or interpolated between knots) afterwards, with $\lambda$ chosen by CV, $C_p$, AIC/BIC.

## Mapping to framework

- **Input:** $X\in\mathbb{R}^{n\times p}$, $y\in\mathbb{R}^n$, link `identity`; mode `lasso`/`lar`.
- **Output:** the **path** $\{\widehat\beta(\lambda)\}$ as a list of knots and equiangular segments.
- **Links:** identity only (least-squares correlations); other links use coordinate-descent GLM paths.
- **Preprocessing:** standardize $X$ to unit norm, center $y$.

## Complexity

- Building/updating $G^{-1}$ via Cholesky up/down-dating: $O(p^2)$ per step plus $O(np)$ for
  correlations; total for the full path $\approx O(p^3 + np^2)$ — comparable to one OLS fit.
- Memory $O(np)$ (or $O(\text{nnz})$ for sparse $X$); the path itself is $O(p\cdot\#\text{knots})$.

## Statistical guarantees

- **Exactness.** With the drop/sign-agreement modification, LARS-Lasso returns the **exact** lasso
  solution path $\widehat\beta(\lambda)$ for all $\lambda\ge 0$ (the homotopy is piecewise linear).
- **Efficiency.** The whole path costs the order of a single least-squares solve, unlike grid-based
  solvers that re-optimize at each $\lambda$.
- Selection/estimation properties are inherited from the lasso (restricted-eigenvalue /
  irrepresentable conditions); see [Lasso via Coordinate Descent](lasso-cd.md).

## Variants & related

- [Lasso via Coordinate Descent](lasso-cd.md) — alternative solver of the same objective;
  produces the same path on a $\lambda$-grid.
- **Forward Stagewise / $\varepsilon$-boosting** — the infinitesimal limit LARS unifies.
- [OLS](ols.md) — the unconstrained endpoint ($\lambda\to 0$, all variables active).
- [Elastic Net](elastic-net.md) — LARS-EN extends the homotopy to the elastic-net penalty.

## References

- <a id="ref-Efron2004lars"></a> <a href="https://doi.org/10.1214/009053604000000067" class="ref-link" target="_blank" rel="noopener noreferrer">Efron, B., Hastie, T., Johnstone, I., and Tibshirani, R. (2004). Least angle regression. *Ann. Statist.*, 32(2):407--499.</a>
- <a id="ref-tibshirani1996regression"></a> <a href="https://doi.org/10.1111/j.2517-6161.1996.tb02080.x" class="ref-link" target="_blank" rel="noopener noreferrer">Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *J. R. Stat. Soc. Ser. B*, 58(1):267--288.</a>
- <a id="ref-buhlmann2011statistics"></a> <a href="https://link.springer.com/book/10.1007/978-3-642-20192-9" class="ref-link" target="_blank" rel="noopener noreferrer">Bühlmann, P. and van de Geer, S. (2011). *Statistics for High-Dimensional Data: Methods, Theory and Applications*. Springer.</a>
