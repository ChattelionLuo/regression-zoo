---
id: debiased-lasso
name: "Debiased / Desparsified Lasso"
aliases: ["desparsified lasso", "de-biased lasso", "LDPE", "low-dimensional projection estimator"]
family: high-dim-inference
regime: [high-dim]
penalty: lasso
link_support: [identity, logit, log]
output: point+inference
year: 2014
refs: [vandegeer2014, Zhang_delasso_2014, javanmard2014confidence, hdi2015, 10.1111/biom.13587, guo2021inference, ma2021global]
status: draft
---

# Debiased / Desparsified Lasso

!!! info "At a glance"
    **Family:** high-dim-inference · **Regime:** high-dim ($p\gg n$) · **Penalty:** lasso (base) ·
    **Output:** point+inference · **Links:** identity, logit, log · **Status:** draft ·
    **Refs:** [`van de Geer et al., 2014`](#ref-vandegeer2014) · [`Zhang and Zhang, 2014`](#ref-Zhang_delasso_2014) · [`Javanmard and Montanari, 2014`](#ref-javanmard2014confidence) · [`Dezeure et al., 2015`](#ref-hdi2015) · [`Xia et al., 2021`](#ref-10.1111/biom.13587)

## Setting & assumptions

- High-dimensional regime: $p$ comparable to or $\gg n$, with sparsity
  $s=\lVert\beta^\star\rVert_0\ll p$. The motivating goal is **inference on individual
  coordinates** $\beta_j^\star$ — confidence intervals and tests — not just point estimation.
- Linear/Gaussian case: $y=X\beta^\star+\varepsilon$, identity link, $\varepsilon$ mean-zero with
  variance $\sigma^2$ (unknown, estimated). GLM extension covers logit/log links via the
  Fisher-information geometry (below).
- The lasso $\widehat\beta$ is consistent in $\ell_1$ but **biased** and non-Gaussian (shrinkage
  toward $0$, no tractable limit law). Debiasing removes the leading bias so a Gaussian limit
  is recovered.
- Design conditions: a restricted-eigenvalue / compatibility condition on
  $\widehat\Sigma=X^\top X/n$, and a **row-sparsity** condition on the precision matrix
  $\Theta^\star=\Sigma^{-1}$ so that $\widehat\Theta$ can be estimated. Sparsity must satisfy
  $s=o\!\big(\sqrt{n}/\log p\big)$ for the remainder term to be asymptotically negligible.

## Estimator / objective

Start from the lasso base estimator
$\widehat\beta=\arg\min_\beta \tfrac1{2n}\lVert y-X\beta\rVert_2^2+\lambda\lVert\beta\rVert_1$.
Its KKT/score residual is biased; the **debiased (desparsified)** estimator adds a single
score-correction step using an approximate inverse $\widehat\Theta$ of
$\widehat\Sigma=X^\top X/n$:

$$
\boxed{\;\widehat b \;=\; \widehat\beta \;+\; \frac{1}{n}\,\widehat\Theta\, X^\top\big(y-X\widehat\beta\big)\;}
$$

This is an **estimating-equation** correction rather than a new penalized objective: it inverts
the lasso KKT conditions so that the $\ell_1$ shrinkage on the target coordinate is removed
("desparsified"). Plugging $y=X\beta^\star+\varepsilon$ gives the bias decomposition

$$
\sqrt{n}\,\big(\widehat b-\beta^\star\big)
=\underbrace{\tfrac{1}{\sqrt n}\,\widehat\Theta X^\top\varepsilon}_{\text{Gaussian term }Z}
\;+\;\underbrace{\sqrt{n}\,\big(I-\widehat\Theta\widehat\Sigma\big)\big(\beta^\star-\widehat\beta\big)}_{\text{remainder }\Delta}.
$$

The Gaussian term satisfies $Z_j\mid X\sim N\!\big(0,\sigma^2[\widehat\Theta\widehat\Sigma\widehat\Theta^\top]_{jj}\big)$.
The remainder is bounded coordinatewise by
$\lVert\Delta\rVert_\infty\le \sqrt{n}\,\lambda_{\text{node}}\,\lVert\widehat\beta-\beta^\star\rVert_1
=O_P\!\big(s\log p/\sqrt n\big)$, which $\to 0$ under $s=o(\sqrt n/\log p)$. Hence each coordinate
is **asymptotically normal and $\sqrt n$-consistent**:

$$
\sqrt{n}\,\big(\widehat b_j-\beta_j^\star\big)\;\xrightarrow{d}\;N\!\big(0,\ \sigma^2\,[\widehat\Theta\widehat\Sigma\widehat\Theta^\top]_{jj}\big),
$$

yielding the $(1-\alpha)$ confidence interval
$\widehat b_j\pm z_{1-\alpha/2}\,\widehat\sigma\sqrt{[\widehat\Theta\widehat\Sigma\widehat\Theta^\top]_{jj}/n}$.

### Constructing $\widehat\Theta$ — nodewise lasso

[`van de Geer et al., 2014`](#ref-vandegeer2014) build $\widehat\Theta$ row by row via **nodewise lasso**: regress each
column $X_{\cdot j}$ on the remaining columns $X_{\cdot\text{-}j}$,

$$
\widehat\gamma_j=\arg\min_{\gamma\in\mathbb{R}^{p-1}}\ \tfrac1{2n}\lVert X_{\cdot j}-X_{\cdot\text{-}j}\gamma\rVert_2^2+\lambda_j\lVert\gamma\rVert_1,
$$

with residual $r_j=X_{\cdot j}-X_{\cdot\text{-}j}\widehat\gamma_j$ and
$\widehat\tau_j^2=r_j^\top X_{\cdot j}/n$. Row $j$ of $\widehat\Theta$ is
$\widehat\Theta_{j\cdot}=\widehat\tau_j^{-2}\,(-\widehat\gamma_{j,1},\dots,1,\dots,-\widehat\gamma_{j,p-1})$ (the
$1$ in position $j$). Equivalently (Javanmard–Montanari) $\widehat\Theta$'s rows solve a
**constrained optimization** minimizing $m^\top\widehat\Sigma m$ subject to
$\lVert\widehat\Sigma m-e_j\rVert_\infty\le\mu$, directly controlling the bias term
$\lVert I-\widehat\Theta\widehat\Sigma\rVert_\infty\le\mu$.

### Variance estimation

The noise level $\sigma^2$ is estimated from the lasso residuals, e.g. the scaled-lasso /
degrees-of-freedom corrected estimator
$\widehat\sigma^2=\lVert y-X\widehat\beta\rVert_2^2/\big(n-\lVert\widehat\beta\rVert_0\big)$ or via
cross-validation; [`Dezeure et al., 2015`](#ref-hdi2015) implements these defaults.

### GLM extension (logit / log links)

For a GLM, replace the Gaussian residual by the score and $\widehat\Sigma$ by the weighted
(Fisher-information) Gram matrix $\widehat\Sigma_W=X^\top \widehat W X/n$ with
$\widehat W=\operatorname{diag}\big(V(\widehat\mu_i)\big)$ evaluated at the penalized MLE $\widehat\beta$.
The debiased estimator becomes

$$
\widehat b=\widehat\beta+\tfrac1n\,\widehat\Theta\,X^\top\big(y-\widehat\mu\big),\qquad
\widehat\Theta\approx\big(X^\top\widehat W X/n\big)^{-1},
$$

with asymptotic variance $[\widehat\Theta\,\widehat\Sigma_W\,\widehat\Theta^\top]_{jj}/n$; the nodewise
regressions are run on the $\sqrt{\widehat W}$-weighted design. [`Xia et al., 2021`](#ref-10.1111/biom.13587)
develop this weighted/Fisher-information correction for GLMs; [`Guo et al., 2021`](#ref-guo2021inference) and
[`Ma et al., 2021`](#ref-ma2021global) give related corrections and (simultaneous) tests for high-dimensional logistic
regression.

## Algorithm

```text
Input: X (n×p), y (n), penalties λ (lasso), {λ_j} (nodewise), link g
1. Fit base lasso:  β̂ = argmin (1/2n)‖y - Xβ‖² + λ‖β‖₁     # GLM: penalized IRLS
2. Build Θ̂ (nodewise lasso), for j = 1..p:
       γ̂_j = argmin (1/2n)‖X[:,j] - X[:,-j]γ‖² + λ_j‖γ‖₁     # GLM: √Ŵ-weighted
       r_j  = X[:,j] - X[:,-j] γ̂_j
       τ̂_j² = r_jᵀ X[:,j] / n
       Θ̂[j,:] = (1/τ̂_j²) · (−γ̂_j with 1 inserted at position j)
3. Debias:   b̂ = β̂ + (1/n) Θ̂ Xᵀ (y - Xβ̂)               # GLM: (y - μ̂), Ŵ-weighted Gram
4. Variance: σ̂²  from lasso residuals;  Ω̂ = Θ̂ Σ̂ Θ̂ᵀ
5. Inference: for each j
       se_j = σ̂ · sqrt( Ω̂_jj / n )
       CI_j = b̂_j ± z_{1-α/2} · se_j
       z-stat = b̂_j / se_j   →  N(0,1) under H₀: β_j = 0
Return  b̂, {se_j}, {CI_j}
```

The Javanmard–Montanari variant replaces step 2 with the per-row constrained QP
$\min_m m^\top\widehat\Sigma m$ s.t. $\lVert\widehat\Sigma m-e_j\rVert_\infty\le\mu$.

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ (base lasso) | CV / $\sqrt{\log p/n}$ | controls $\widehat\beta$; bias of $\widehat b$ is first-order insensitive to it |
| $\lambda_j$ (nodewise) | CV per column | governs $\widehat\Theta$ quality and the remainder $\lVert I-\widehat\Theta\widehat\Sigma\rVert_\infty$ |
| $\mu$ (JM variant) | $\asymp\sqrt{\log p/n}$ | bias bound in the constrained-QP construction |
| $\widehat\sigma^2$ | scaled lasso / df-corrected | noise level for Gaussian SEs |
| link / weights | identity | logit, log via Fisher weights $\widehat W$ (GLM extension) |
| $\alpha$ | $0.05$ | CI level; Bonferroni/holm or `ma2021global` for simultaneity |

## Mapping to framework

- **Input:** $X\in\mathbb{R}^{n\times p}$, $y\in\mathbb{R}^n$, link $g$, penalties $\lambda,\{\lambda_j\}$.
- **Output:** debiased point estimate $\widehat b$ **plus inference** — standard errors $\{se_j\}$,
  $z$-statistics/$p$-values, and CIs $\{CI_j\}$ (hence `output: point+inference`).
- **Links:** identity (Gaussian); logit, log via the weighted Fisher-information correction.
- **Preprocessing:** standardize columns of $X$; center $y$ (Gaussian) or fit unpenalized
  intercept (GLM). Coefficients/SEs returned on the requested scale.

## Complexity

- Base lasso: as in [Lasso via Coordinate Descent](lasso-cd.md), $O(npL)$ over a path.
- Nodewise $\widehat\Theta$: **$p$ separate lasso regressions**, each $O(np)$ per cycle ⇒
  dominant cost $O(np^2)$ (embarrassingly parallel over columns).
- Debias + variance: $O(np)$ for the score correction; forming the needed diagonal of
  $\widehat\Theta\widehat\Sigma\widehat\Theta^\top$ is $O(np)$ per coordinate.
- Memory $O(np)$ (rows of $\widehat\Theta$ can be streamed/discarded after extracting $se_j$).

## Statistical guarantees

- **Asymptotic normality:** under compatibility + precision-row-sparsity and
  $s=o(\sqrt n/\log p)$, $\sqrt n(\widehat b_j-\beta_j^\star)\xrightarrow{d}N(0,\sigma^2\Omega_{jj})$
  with $\Omega=\Theta\Sigma\Theta^\top$ ([`van de Geer et al., 2014`](#ref-vandegeer2014), [`Zhang and Zhang, 2014`](#ref-Zhang_delasso_2014),
  [`Javanmard and Montanari, 2014`](#ref-javanmard2014confidence)).
- **Coverage / honesty:** the resulting CIs achieve asymptotically nominal coverage uniformly
  over the sparse parameter space; the remainder term is uniformly negligible.
- **Efficiency:** the asymptotic variance attains the semiparametric efficiency bound under the
  stated conditions [`Janková and van de Geer, 2018`](#ref-jankova2018semiparametric).
- **GLM:** analogous normality with the Fisher-information variance for logit/log links
  ([`Xia et al., 2021`](#ref-10.1111/biom.13587), [`Guo et al., 2021`](#ref-guo2021inference)); [`Ma et al., 2021`](#ref-ma2021global) provides simultaneous/global tests.

## Variants & related

- [Lasso via Coordinate Descent](lasso-cd.md) — the base estimator that is debiased here.
- [Decorrelated Score (Ning–Liu)](decorrelated-score.md) — score-based inference framework with
  the same orthogonalization idea, stated directly for general GLMs.
- **Nodewise lasso** (Meinshausen–Bühlmann) — the precision-matrix construction reused for $\widehat\Theta$.
- **Scaled / square-root lasso** — pivotal noise-level estimation feeding the SEs.

## References

- <a id="ref-vandegeer2014"></a> <a href="https://doi.org/10.1214/14-AOS1221" class="ref-link" target="_blank" rel="noopener noreferrer">van de Geer, S., Bühlmann, P., Ritov, Y. A., and Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Ann. Statist.*, 42(3):1166--1202.</a>
- <a id="ref-Zhang_delasso_2014"></a> <a href="https://doi.org/10.1111/rssb.12026" class="ref-link" target="_blank" rel="noopener noreferrer">Zhang, C.-H. and Zhang, S. S. (2014). Confidence intervals for low dimensional parameters in high dimensional linear models. *J. R. Stat. Soc. Ser. B*, 76(1):217--242.</a>
- <a id="ref-javanmard2014confidence"></a> <a href="https://jmlr.org/papers/v15/javanmard14a.html" class="ref-link" target="_blank" rel="noopener noreferrer">Javanmard, A. and Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *J. Mach. Learn. Res.*, 15(1):2869--2909.</a>
- <a id="ref-hdi2015"></a> <a href="https://doi.org/10.1214/15-STS527" class="ref-link" target="_blank" rel="noopener noreferrer">Dezeure, R., Bühlmann, P., Meier, L., and Meinshausen, N. (2015). High-dimensional inference: confidence intervals, p-values and R-software hdi. *Stat. Sci.*, 30(4):533--558.</a>
- <a id="ref-10.1111/biom.13587"></a> <a href="https://doi.org/10.1111/biom.13587" class="ref-link" target="_blank" rel="noopener noreferrer">Xia, L., Nan, B., and Li, Y. (2021). Debiased Lasso for generalized linear models with a diverging number of covariates. *Biometrics*, 79(1):344--357.</a>
- <a id="ref-guo2021inference"></a> <a href="https://jmlr.org/papers/v22/19-1149.html" class="ref-link" target="_blank" rel="noopener noreferrer">Guo, Z., Rakshit, P., Herman, D. S., and Chen, J. (2021). Inference for the case probability in high-dimensional logistic regression. *J. Mach. Learn. Res.*, 22(254):1--54.</a>
- <a id="ref-ma2021global"></a> <a href="https://doi.org/10.1080/01621459.2020.1770098" class="ref-link" target="_blank" rel="noopener noreferrer">Ma, R., Cai, T. T., and Li, H. (2021). Global and simultaneous hypothesis testing for high-dimensional logistic regression models. *J. Amer. Statist. Assoc.*, 116(534):984--998.</a>
- <a id="ref-jankova2018semiparametric"></a> <a href="https://doi.org/10.1214/17-AOS1622" class="ref-link" target="_blank" rel="noopener noreferrer">Janková, J. and van de Geer, S. (2018). Semiparametric efficiency bounds for high-dimensional models. *Ann. Statist.*, 46(5):2336--2359.</a>
