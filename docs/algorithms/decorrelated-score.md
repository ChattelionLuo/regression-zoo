---
id: decorrelated-score
name: "Decorrelated Score (Ning–Liu)"
aliases: ["decorrelated score test", "orthogonalized score", "Ning-Liu score"]
family: high-dim-inference
regime: [high-dim]
penalty: lasso
link_support: [any]
output: point+inference
year: 2017
refs: [10.1214/16-AOS1448, vandegeer2014, javanmard2014confidence, jankova2018semiparametric]
status: draft
---

# Decorrelated Score (Ning–Liu)

!!! info "At a glance"
    **Family:** high-dim-inference · **Regime:** high-dim ($p\gg n$) · **Penalty:** lasso (base) ·
    **Output:** point+inference · **Links:** any · **Status:** draft ·
    **Refs:** [`Ning and Liu, 2017`](#ref-10.1214/16-AOS1448) · [`van de Geer et al., 2014`](#ref-vandegeer2014) · [`Javanmard and Montanari, 2014`](#ref-javanmard2014confidence)

## Setting & assumptions

- High-dimensional model with a **low-dimensional target**: partition the coefficient
  $\beta=(\theta,\gamma)$ where $\theta\in\mathbb{R}$ (or low-dimensional) is the parameter of
  interest and $\gamma\in\mathbb{R}^{p-1}$ is a high-dimensional **nuisance**. The aim is a valid
  test of $H_0:\theta=\theta_0$ and a CI for $\theta$.
- Any twice-differentiable likelihood in the [exponential family](../framework/notation.md#2-exponential-dispersion-family):
  the framework is stated generically through the loss $\mathcal L(\beta)$, its score
  $\nabla\mathcal L$ and Fisher information $I$, so it applies across GLMs (linear, logistic,
  Poisson, …) — hence `link_support: [any]`.
- Sparsity $\lVert\gamma^\star\rVert_0\ll p$ and a compatibility/restricted-eigenvalue condition,
  so a penalized (lasso) estimator $\widehat\gamma$ is $\ell_1$-consistent. The nuisance score block
  $I_{\gamma\gamma}$ has a sparse inverse direction, enabling the decorrelation step.
- **Key difficulty:** the ordinary score $\nabla_\theta\mathcal L$ is sensitive to the
  estimation error in $\widehat\gamma$ (its bias does not vanish at $\sqrt n$ rate). The decorrelated
  score removes this first-order sensitivity.

## Estimator / objective

Write the partitioned information at $\beta^\star$,
$I=\begin{psmallmatrix}I_{\theta\theta}&I_{\theta\gamma}\\ I_{\gamma\theta}&I_{\gamma\gamma}\end{psmallmatrix}=\nabla^2\mathcal L$.
The **decorrelated (orthogonalized) score** projects the target score orthogonal to the nuisance
directions:

$$
\boxed{\;S(\theta,\gamma)=\nabla_\theta\mathcal L(\theta,\gamma)-w^\top\nabla_\gamma\mathcal L(\theta,\gamma),\qquad
w^\top=I_{\theta\gamma}\,I_{\gamma\gamma}^{-1}\;}
$$

The weight $w$ is exactly the projection coefficient of the target score onto the nuisance score
space, so that $\mathbb{E}\big[S\cdot\nabla_\gamma\mathcal L\big]=0$: $S$ is **orthogonal to the
nuisance**, making it insensitive to estimation error in $\gamma$. The information for $\theta$
after removing the nuisance is the partial (efficient) information

$$
I_{\theta\mid\gamma}=I_{\theta\theta}-I_{\theta\gamma}I_{\gamma\gamma}^{-1}I_{\gamma\theta}.
$$

In practice $w$ is estimated by a **sparse / Dantzig-type regression** of the target-score
components on the nuisance-score components (equivalently a penalized solve of
$I_{\gamma\gamma}\,w=I_{\gamma\theta}$):

$$
\widehat w=\arg\min_{w}\ \lVert w\rVert_1\quad\text{s.t.}\quad
\big\lVert \widehat I_{\gamma\gamma}\,w-\widehat I_{\gamma\theta}\big\rVert_\infty\le\mu,
$$

with $\widehat I=\nabla^2\mathcal L(\widehat\theta_0,\widehat\gamma)$ evaluated at the penalized estimator.
The plug-in **decorrelated score statistic** for $H_0:\theta=\theta_0$ is then

$$
\widehat S=\nabla_\theta\mathcal L(\theta_0,\widehat\gamma)-\widehat w^\top\nabla_\gamma\mathcal L(\theta_0,\widehat\gamma),
\qquad
U_n=\frac{\sqrt n\,\widehat S}{\sqrt{\widehat I_{\theta\mid\gamma}}}\ \xrightarrow{d}\ N(0,1)\ \text{ under }H_0,
$$

and $U_n^2\xrightarrow{d}\chi^2_1$ (more generally $\chi^2_d$ for a $d$-dimensional target). A
one-step / point estimator is recovered by a Newton correction along the decorrelated direction,

$$
\widehat\theta=\theta_0-\frac{\widehat S}{\widehat I_{\theta\mid\gamma}},\qquad
\sqrt n(\widehat\theta-\theta^\star)\xrightarrow{d}N\!\big(0,\,I_{\theta\mid\gamma}^{-1}\big).
$$

**Inverting for CIs.** Collect all $\theta_0$ not rejected at level $\alpha$:

$$
\mathcal C_{1-\alpha}=\Big\{\theta_0:\ |U_n(\theta_0)|\le z_{1-\alpha/2}\Big\}
=\ \widehat\theta\ \pm\ z_{1-\alpha/2}\,\big(n\,\widehat I_{\theta\mid\gamma}\big)^{-1/2}.
$$

## Algorithm

```text
Input: X, y, loss L (GLM neg-loglik), target index θ, null θ₀, penalties λ, μ
1. Penalized nuisance fit:
       γ̂ = argmin_γ  L(θ₀, γ) + λ‖γ‖₁          # lasso/penalized MLE with θ fixed at θ₀
2. Plug-in information at (θ₀, γ̂):
       Î = ∇²L(θ₀, γ̂)  →  blocks Î_θθ, Î_θγ, Î_γγ
3. Decorrelation weight (sparse / Dantzig solve):
       ŵ = argmin ‖w‖₁  s.t.  ‖Î_γγ w − Î_γθ‖_∞ ≤ μ
4. Decorrelated score and partial information:
       Ŝ = ∇_θ L(θ₀, γ̂) − ŵᵀ ∇_γ L(θ₀, γ̂)
       Î_{θ|γ} = Î_θθ − ŵᵀ Î_γθ
5. Test statistic:
       U = √n · Ŝ / sqrt(Î_{θ|γ})          # N(0,1) under H₀;  U² ~ χ²₁
6. One-step estimate & CI:
       θ̂   = θ₀ − Ŝ / Î_{θ|γ}
       CI  = θ̂ ± z_{1-α/2} / sqrt(n · Î_{θ|γ})
Return  θ̂, U (p-value), CI
```

## Hyperparameters & configuration

| Knob | Default | Notes |
|---|---|---|
| $\lambda$ (nuisance lasso) | CV / $\sqrt{\log p/n}$ | regularizes $\widehat\gamma$; the score is first-order insensitive to its error |
| $\mu$ (Dantzig tol) | $\asymp\sqrt{\log p/n}$ | controls bias of the decorrelation weight $\widehat w$ |
| loss / link | any GLM | identity, logit, log, … via $\nabla\mathcal L,\nabla^2\mathcal L$ |
| target dim $d$ | $1$ | $\chi^2_d$ null for multi-dimensional $\theta$ |
| $\alpha$ | $0.05$ | test level / CI coverage |

## Mapping to framework

- **Input:** $X\in\mathbb{R}^{n\times p}$, $y\in\mathbb{R}^n$, link $g$, target coordinate(s)
  $\theta$, null value $\theta_0$, penalties $\lambda,\mu$.
- **Output:** point estimate $\widehat\theta$ **plus inference** — the statistic $U_n$
  ($p$-value) and CI $\mathcal C_{1-\alpha}$ (hence `output: point+inference`).
- **Links:** any — the construction only uses the loss gradient/Hessian
  $\nabla\mathcal L,\ \nabla^2\mathcal L=I$.
- **Preprocessing:** standardize $X$; unpenalized intercept as usual. The nuisance partition is
  fixed by the user (which coordinate is being tested).

## Complexity

- Step 1 (penalized nuisance fit): one lasso/penalized-IRLS solve, $O(np)$ per cycle.
- Step 3 (decorrelation weight): one sparse/Dantzig regression of dimension $p-1$, $O(np)$ per
  cycle (or an LP for the Dantzig form).
- Steps 4–6: $O(np)$ for the score and partial information; testing a single target is cheap.
- Testing many coordinates ⇒ repeat the decorrelation per target (parallelizable), dominant cost
  $O(np^2)$ overall; memory $O(np)$.

## Statistical guarantees

- **Asymptotic null distribution:** under compatibility + sparsity
  ($s\log p/\sqrt n\to 0$) and a valid sparse $\widehat w$,
  $U_n\xrightarrow{d}N(0,1)$ under $H_0$ and $U_n^2\xrightarrow{d}\chi^2_d$
  [`Ning and Liu, 2017`](#ref-10.1214/16-AOS1448).
- **Estimation / normality:** the one-step estimator is $\sqrt n$-consistent and asymptotically
  normal with variance $I_{\theta\mid\gamma}^{-1}$, the semiparametric efficiency bound for
  $\theta$ in the presence of the nuisance [`Janková and van de Geer, 2018`](#ref-jankova2018semiparametric).
- **Honest coverage:** inverting the test gives CIs with asymptotically nominal coverage,
  uniformly over the sparse nuisance space; validity is **robust to model misspecification** of
  the nuisance and to moderate nuisance estimation error.
- The construction generalizes the desparsified-lasso normal limit ([`van de Geer et al., 2014`](#ref-vandegeer2014),
  [`Javanmard and Montanari, 2014`](#ref-javanmard2014confidence)) from linear models to general (penalized) likelihoods.

## Variants & related

- [Debiased / Desparsified Lasso](debiased-lasso.md) — the linear-model special case; its
  $\widehat\Theta X^\top$ correction coincides with decorrelation under Gaussian loss.
- [Lasso via Coordinate Descent](lasso-cd.md) — supplies the penalized nuisance estimate $\widehat\gamma$.
- **Double / orthogonalized (Neyman) score** — the same orthogonality principle underlies
  double-ML and partialling-out estimators.

## References

- <a id="ref-10.1214/16-AOS1448"></a> Ning, Y. and Liu, H. (2017). A general theory of hypothesis tests and confidence regions for sparse high dimensional models. *Ann. Statist.*, 45(1):158--195.
- <a id="ref-vandegeer2014"></a> van de Geer, S., Bühlmann, P., Ritov, Y. A., and Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Ann. Statist.*, 42(3):1166--1202.
- <a id="ref-javanmard2014confidence"></a> Javanmard, A. and Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *J. Mach. Learn. Res.*, 15(1):2869--2909.
- <a id="ref-jankova2018semiparametric"></a> Janková, J. and van de Geer, S. (2018). Semiparametric efficiency bounds for high-dimensional models. *Ann. Statist.*, 46(5):2336--2359.
