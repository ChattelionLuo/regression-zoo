# Unified Notation & Mathematical Framework

Every algorithm card in this encyclopedia answers the **same question** in the **same language**:

> Given a design matrix $X$, a response $y$, and a link function $g$, produce a coefficient
> estimate $\hat\beta$ — under a stated configuration of hyperparameters.

This page fixes the notation and the canonical objects (exponential family, link, loss,
gradient, Hessian, penalties, optimization template). **All cards reference this page** so
that estimators defined in different papers can be compared on equal footing.

---

## 1. Data

| Symbol | Meaning |
|---|---|
| $n$ | number of observations |
| $p$ | number of covariates (features) |
| $X \in \mathbb{R}^{n\times p}$ | design / covariate matrix, rows $x_i^\top$, columns $X_{\cdot j}$ |
| $x_i \in \mathbb{R}^{p}$ | covariate vector for observation $i$ |
| $y \in \mathbb{R}^{n}$ | response vector, entries $y_i$ |
| $\beta \in \mathbb{R}^{p}$ | unknown coefficient vector; $\beta^\star$ true value, $\hat\beta$ estimate |
| $\beta_0$ | optional intercept (absorbed into $X$ via a column of ones unless stated) |

**Standardization.** Unless a card says otherwise, columns of $X$ are centered and scaled to
unit $\ell_2$-norm (or unit variance), and $y$ is centered, so the intercept can be handled
separately. Cards that depend on a specific scaling state it explicitly.

**Regimes.** We label each algorithm's intended data regime:

- **low-dimensional**: $n \gg p$, $X$ typically full column rank.
- **high-dimensional**: $p$ comparable to or $\gg n$; sparsity ($s = \lVert\beta^\star\rVert_0 \ll p$) is usually assumed.
- **streaming / online**: data arrive sequentially in points or batches; storage/compute per step is bounded.

---

## 2. Exponential dispersion family

Responses follow a one-parameter exponential dispersion family with natural parameter
$\theta_i$ and dispersion $\phi$:

$$
f(y_i; \theta_i, \phi) = \exp\!\left\{ \frac{y_i \theta_i - b(\theta_i)}{a(\phi)} + c(y_i, \phi) \right\}.
$$

Key identities:

$$
\mu_i := \mathbb{E}[y_i] = b'(\theta_i), \qquad
\operatorname{Var}(y_i) = a(\phi)\, b''(\theta_i) = a(\phi)\, V(\mu_i),
$$

where $V(\mu) = b''(\theta(\mu))$ is the **variance function**.

| Family | $b(\theta)$ | $\mu(\theta)$ | $V(\mu)$ | dispersion $a(\phi)$ |
|---|---|---|---|---|
| Gaussian | $\theta^2/2$ | $\theta$ | $1$ | $\sigma^2$ |
| Bernoulli / Binomial | $\log(1+e^{\theta})$ | $\tfrac{e^\theta}{1+e^\theta}$ | $\mu(1-\mu)$ | $1$ |
| Poisson | $e^{\theta}$ | $e^{\theta}$ | $\mu$ | $1$ |
| Gamma | $-\log(-\theta)$ | $-1/\theta$ | $\mu^2$ | $\nu^{-1}$ |

---

## 3. Link function

The systematic component connects the mean to a **linear predictor**:

$$
\eta_i = x_i^\top \beta, \qquad g(\mu_i) = \eta_i \iff \mu_i = g^{-1}(\eta_i).
$$

$g$ is the **link function** (monotone, differentiable). The **canonical link** is the one for
which $\theta_i = \eta_i$, i.e. $g = (b')^{-1}$.

| Family | Canonical link $g(\mu)$ | Inverse link $g^{-1}(\eta)$ |
|---|---|---|
| Gaussian | identity $\mu$ | $\eta$ |
| Bernoulli | logit $\log\frac{\mu}{1-\mu}$ | $\frac{1}{1+e^{-\eta}}$ |
| Poisson | log $\log\mu$ | $e^{\eta}$ |
| Gamma | inverse $-1/\mu$ (or reciprocal) | $-1/\eta$ |

Cards state which links they support (often: canonical only, or any differentiable $g$).

---

## 4. Loss / objective

We write the per-sample **negative log-likelihood** (up to constants) as $\ell_i(\beta)$ and the
empirical loss as

$$
\mathcal{L}(\beta) = \frac{1}{n}\sum_{i=1}^{n} \ell_i(\beta).
$$

For a canonical-link GLM,

$$
\ell_i(\beta) = -\,\big[\, y_i\, x_i^\top\beta - b(x_i^\top\beta) \,\big] \big/ a(\phi),
$$

which specializes to the familiar losses (dropping constant factors):

$$
\textbf{OLS:}\quad \mathcal{L}(\beta) = \frac{1}{2n}\lVert y - X\beta\rVert_2^2,
$$
$$
\textbf{Logistic:}\quad \mathcal{L}(\beta) = \frac{1}{n}\sum_{i=1}^n \Big[\log\!\big(1+e^{x_i^\top\beta}\big) - y_i\, x_i^\top\beta\Big],
$$
$$
\textbf{Poisson:}\quad \mathcal{L}(\beta) = \frac{1}{n}\sum_{i=1}^n \Big[ e^{x_i^\top\beta} - y_i\, x_i^\top\beta \Big].
$$

### Gradient and Hessian (canonical link)

With $\mu(\beta) = \big(g^{-1}(x_1^\top\beta),\dots\big)^\top$ and $W(\beta) = \operatorname{diag}\big(V(\mu_i)\big)$,

$$
\nabla \mathcal{L}(\beta) = -\frac{1}{n} X^\top \big(y - \mu(\beta)\big),
\qquad
\nabla^2 \mathcal{L}(\beta) = \frac{1}{n} X^\top W(\beta)\, X.
$$

The quantity $S(\beta) = X^\top(y-\mu(\beta))$ is the **score**; $I(\beta)=X^\top W X$ the
(observed/expected, equal under canonical link) **Fisher information**.

---

## 5. The estimator template

The overwhelming majority of GLM solvers compute

$$
\boxed{\;\hat\beta \in \arg\min_{\beta\in\mathbb{R}^p}\; \mathcal{L}(\beta) \;+\; \lambda\, P(\beta)\;}
$$

where $P$ is a (possibly zero) penalty and $\lambda \ge 0$ a regularization level. Cards describe
**how** the minimization is performed (closed form, Newton/IRLS, coordinate descent, proximal/
first-order, homotopy/path, stochastic/online, ...), which is precisely the "algorithm".

Some cards instead define $\hat\beta$ through an **estimating equation** $\sum_i \psi_i(\beta)=0$
(e.g. GEE, quasi-likelihood, debiasing corrections); the same notation applies.

### Common penalties $P(\beta)$

| Name | $P(\beta)$ | Notes |
|---|---|---|
| Ridge ($\ell_2$) | $\tfrac12\lVert\beta\rVert_2^2$ | strongly convex, no sparsity |
| Lasso ($\ell_1$) | $\lVert\beta\rVert_1$ | convex, sparse |
| Elastic net | $\alpha\lVert\beta\rVert_1 + \tfrac{1-\alpha}{2}\lVert\beta\rVert_2^2$ | $\alpha\in[0,1]$ |
| SCAD | nonconvex, see card | unbiased for large $|\beta_j|$ |
| MCP | nonconvex, see card | minimax concave |
| Group lasso | $\sum_{g} \sqrt{d_g}\,\lVert\beta_g\rVert_2$ | group sparsity |
| Fused lasso | $\lVert\beta\rVert_1 + \sum_j |\beta_j-\beta_{j-1}|$ | piecewise constant |
| Adaptive lasso | $\sum_j w_j |\beta_j|$ | data-driven weights $w_j$ |

### Proximal operators (used by first-order solvers)

$$
\operatorname{prox}_{t P}(v) = \arg\min_{\beta}\; \tfrac{1}{2t}\lVert \beta - v\rVert_2^2 + P(\beta).
$$

Soft-thresholding (lasso prox): $\;\operatorname{prox}_{t\lambda\lVert\cdot\rVert_1}(v)_j = \operatorname{sign}(v_j)\,(|v_j|-t\lambda)_+ =: \mathcal{S}_{t\lambda}(v_j).$

---

## 6. Optimization primitives (shared vocabulary)

Cards reuse these building blocks rather than redefining them:

- **IRLS / Fisher scoring.** Iterate $\beta^{(t+1)} = (X^\top W X)^{-1} X^\top W z$, with working
  response $z_i = \eta_i + (y_i-\mu_i)\,g'(\mu_i)$ and weights $W = \operatorname{diag}(1/(g'(\mu_i)^2 V(\mu_i)))$.
- **Newton / proximal-Newton.** Use $\nabla\mathcal L$, $\nabla^2\mathcal L$; for penalized problems
  solve a penalized quadratic each step.
- **Coordinate descent.** Cyclically minimize over each $\beta_j$ (closed-form with soft-thresholding for $\ell_1$).
- **(F)ISTA / proximal gradient.** $\beta^{(t+1)} = \operatorname{prox}_{t\lambda P}\!\big(\beta^{(t)} - t\,\nabla\mathcal L(\beta^{(t)})\big)$.
- **Homotopy / LARS.** Track the exact solution path as $\lambda$ varies.
- **Stochastic gradient (SGD).** $\beta^{(t+1)} = \beta^{(t)} - \gamma_t\, \nabla \ell_{i_t}(\beta^{(t)})$ (+ optional prox).
- **ADMM.** Split $\mathcal L + \lambda P$ via an auxiliary variable and dual updates.

---

## 7. Evaluation interface (for Phase 2/3)

Every implemented solver will expose the same contract:

```text
fit(X, y, link, **config)  ->  beta_hat   (plus optional path, diagnostics)
```

so that, given identical $(X, y, g)$, different cards' outputs $\hat\beta$ can be compared,
correlated, and benchmarked across datasets. The card's **"Mapping to framework"** section is
the spec for this interface.

---

## 8. Symbol glossary (quick reference)

| Symbol | Meaning |
|---|---|
| $\eta = X\beta$ | linear predictor |
| $\mu = g^{-1}(\eta)$ | mean response |
| $g, g^{-1}$ | link, inverse link |
| $b(\cdot), V(\cdot)$ | cumulant, variance function |
| $\mathcal L(\beta)$ | empirical loss (mean neg. log-likelihood) |
| $S(\beta), I(\beta)$ | score, Fisher information |
| $P(\beta), \lambda$ | penalty, regularization level |
| $\mathcal S_t(\cdot)$ | soft-threshold at level $t$ |
| $\operatorname{prox}_{tP}$ | proximal operator of $tP$ |
| $\gamma_t$ | (stochastic) step size / learning rate |
| $s$ | sparsity level $\lVert\beta^\star\rVert_0$ |
