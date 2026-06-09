<p class="arena-kicker">Phase II · Comparative study</p>

# The Arena

Every solver from Phase I has a faithful Python implementation and has been run across **16 datasets**, real and synthetic, regression and logistic, with up to 50 hyperparameter configurations per solver where applicable.

Each *(solver, config)* run produces one coefficient vector $\widehat\beta \in \mathbb{R}^p$.  The two visualisations below make those vectors comparable:

- **Embedding map**: unit-normalised vectors $\widehat\beta/\|\widehat\beta\|$ projected to 2D by t-SNE and UMAP.  Algorithmically similar solutions cluster; hover to see the solver name and exact config.
- **Coefficient heatmap**: all runs for one dataset as a matrix (rows = runs, columns = features $j$). The top bar shows mean $|\widehat\beta_j|$ per feature. Colour = $\widehat\beta_j$ on a diverging RdBu scale. Hover for exact value.

All solvers share the interface `SolverCls(config={...}).fit(X, y, link=...) -> FitResult`.

---

## Datasets

| Dataset | Source | n | p | Task |
|---|---|---|---|---|
| Diabetes | UCI / sklearn | 442 | 10 | Regression |
| Breast Cancer | UCI / sklearn | 569 | 30 | Logistic |
| Digits (>= 5) | NIST / sklearn | 1 797 | 64 | Logistic |
| Fair Affairs | statsmodels | 2 000 | 8 | Regression |
| RAND HIE | statsmodels | 2 000 | 9 | Regression |
| STAR98 | statsmodels | 303 | 21 | Logistic |
| ANES96 | statsmodels | 944 | 10 | Logistic |
| Mode Choice | statsmodels | 840 | 8 | Logistic |
| Synth sparse (p=200) | synthetic | 2 000 | 200 | Regression |
| Synth dense (p=100) | synthetic | 2 000 | 100 | Regression |
| Synth correlated (p=150) | synthetic | 2 000 | 150 | Regression |
| Synth high-dim (p=300) | synthetic | 2 000 | 300 | Regression |
| Synth Friedman #1 | synthetic | 2 000 | 10 | Regression |
| Synth logit dense (p=100) | synthetic | 2 000 | 100 | Logistic |
| Synth logit sparse (p=200) | synthetic | 2 000 | 200 | Logistic |
| Synth logit noisy (p=150) | synthetic | 2 000 | 150 | Logistic |

Real datasets are standardised (zero mean, unit variance). Synthetic datasets are generated with controlled sparsity, rank structure, and noise level.

