<p class="arena-kicker">Phase II · Comparative study</p>

# The Arena

Every solver from Phase I has a faithful Python implementation and has been run across **16 datasets** — real and synthetic, regression and logistic — with up to 50 hyperparameter configurations per solver where applicable.

Each *(solver, config)* run produces one coefficient vector $\hat\beta \in \mathbb{R}^p$.  The two visualisations below make those vectors comparable:

- **Embedding map** — unit-normalised vectors $\hat\beta/\|\hat\beta\|$ projected to 2D by t-SNE and UMAP.  Algorithmically similar solutions cluster; hover to see the solver name and exact config.
- **Coefficient heatmap** — all runs for one dataset as a matrix (rows = runs, columns = features $j$). Colour = $\hat\beta_j$ on a diverging RdBu scale. Which features does each family activate? How does regularisation reshape the pattern?

All solvers share the interface `SolverCls(config={...}).fit(X, y, link=...) → FitResult`.

---

## Real datasets

### Diabetes  <small>· n = 442, p = 10 · Regression · UCI / sklearn</small>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Diabetes.html" class="arena-embed-frame" title="Embedding: Diabetes"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Diabetes.html" class="arena-heatmap-frame" title="Heatmap: Diabetes"></iframe>
</div>

### Breast Cancer  <small>· n = 569, p = 30 · Logistic · UCI / sklearn</small>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Breast_Cancer.html" class="arena-embed-frame" title="Embedding: Breast Cancer"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Breast_Cancer.html" class="arena-heatmap-frame" title="Heatmap: Breast Cancer"></iframe>
</div>

### Digits (≥ 5)  <small>· n = 1 797, p = 64 · Logistic · NIST / sklearn</small>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Digits_hi5.html" class="arena-embed-frame" title="Embedding: Digits"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Digits_hi5.html" class="arena-heatmap-frame" title="Heatmap: Digits"></iframe>
</div>

### Fair Affairs  <small>· n = 2 000, p = 8 · Regression · statsmodels</small>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Fair_Affairs.html" class="arena-embed-frame" title="Embedding: Fair Affairs"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Fair_Affairs.html" class="arena-heatmap-frame" title="Heatmap: Fair Affairs"></iframe>
</div>

### RAND HIE  <small>· n = 2 000, p = 9 · Regression · statsmodels</small>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_RAND_HIE.html" class="arena-embed-frame" title="Embedding: RAND HIE"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_RAND_HIE.html" class="arena-heatmap-frame" title="Heatmap: RAND HIE"></iframe>
</div>

### STAR98 pass rate  <small>· n = 303, p = 21 · Logistic · statsmodels</small>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_STAR98_pass.html" class="arena-embed-frame" title="Embedding: STAR98"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_STAR98_pass.html" class="arena-heatmap-frame" title="Heatmap: STAR98"></iframe>
</div>

### ANES96 vote  <small>· n = 944, p = 10 · Logistic · statsmodels</small>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_ANES96_vote.html" class="arena-embed-frame" title="Embedding: ANES96"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_ANES96_vote.html" class="arena-heatmap-frame" title="Heatmap: ANES96"></iframe>
</div>

### Mode Choice  <small>· n = 840, p = 8 · Logistic · statsmodels</small>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Mode_Choice.html" class="arena-embed-frame" title="Embedding: Mode Choice"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Mode_Choice.html" class="arena-heatmap-frame" title="Heatmap: Mode Choice"></iframe>
</div>

---

## Synthetic — regression

### Sparse (p = 200, 20 informative)

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Reg_Sparse200.html" class="arena-embed-frame" title="Embedding: Sparse 200"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Reg_Sparse200.html" class="arena-heatmap-frame" title="Heatmap: Sparse 200"></iframe>
</div>

### Dense (p = 100, all informative)

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Reg_Dense100.html" class="arena-embed-frame" title="Embedding: Dense 100"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Reg_Dense100.html" class="arena-heatmap-frame" title="Heatmap: Dense 100"></iframe>
</div>

### Correlated (p = 150, effective rank 10)

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Reg_Corr150.html" class="arena-embed-frame" title="Embedding: Correlated 150"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Reg_Corr150.html" class="arena-heatmap-frame" title="Heatmap: Correlated 150"></iframe>
</div>

### Very high-dimensional (p = 300, 30 informative)

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Reg_HighDim300.html" class="arena-embed-frame" title="Embedding: HighDim 300"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Reg_HighDim300.html" class="arena-heatmap-frame" title="Heatmap: HighDim 300"></iframe>
</div>

### Friedman \#1  (p = 10, nonlinear ground truth)

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Friedman1.html" class="arena-embed-frame" title="Embedding: Friedman1"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Friedman1.html" class="arena-heatmap-frame" title="Heatmap: Friedman1"></iframe>
</div>

---

## Synthetic — logistic

### Dense (p = 100, 80 informative)

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Logit_Dense100.html" class="arena-embed-frame" title="Embedding: Logit Dense100"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Logit_Dense100.html" class="arena-heatmap-frame" title="Heatmap: Logit Dense100"></iframe>
</div>

### Sparse (p = 200, 25 informative)

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Logit_Sparse200.html" class="arena-embed-frame" title="Embedding: Logit Sparse200"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Logit_Sparse200.html" class="arena-heatmap-frame" title="Heatmap: Logit Sparse200"></iframe>
</div>

### Noisy (p = 150, 12 % label noise)

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Logit_Noisy150.html" class="arena-embed-frame" title="Embedding: Logit Noisy150"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Logit_Noisy150.html" class="arena-heatmap-frame" title="Heatmap: Logit Noisy150"></iframe>
</div>

---

## Datasets

16 datasets spanning real-world and synthetic regression and logistic tasks.

| Dataset | Source | n | p | Task |
|---|---|---|---|---|
| Diabetes | UCI / sklearn | 442 | 10 | Regression |
| Breast Cancer | UCI / sklearn | 569 | 30 | Logistic |
| Digits (≥ 5) | NIST / sklearn | 1 797 | 64 | Logistic |
| Fair Affairs | statsmodels | 2 000 | 8 | Regression |
| RAND HIE | statsmodels | 2 000 | 9 | Regression |
| STAR98 pass | statsmodels | 303 | 21 | Logistic |
| ANES96 vote | statsmodels | 944 | 10 | Logistic |
| Mode Choice | statsmodels | 840 | 8 | Logistic |
| Synth — sparse (p=200) | synthetic | 2 000 | 200 | Regression |
| Synth — dense (p=100) | synthetic | 2 000 | 100 | Regression |
| Synth — correlated (p=150) | synthetic | 2 000 | 150 | Regression |
| Synth — high-dim (p=300) | synthetic | 2 000 | 300 | Regression |
| Synth — Friedman \#1 | synthetic | 2 000 | 10 | Regression |
| Synth logit — dense (p=100) | synthetic | 2 000 | 100 | Logistic |
| Synth logit — sparse (p=200) | synthetic | 2 000 | 200 | Logistic |
| Synth logit — noisy (p=150) | synthetic | 2 000 | 150 | Logistic |

Real datasets (sklearn, statsmodels) are standardised; features are scaled to zero mean and unit variance. Synthetic datasets are generated with controlled sparsity, rank structure, and noise level.

