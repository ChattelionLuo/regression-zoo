<p class="arena-kicker">Phase II 繚 Comparative study</p>

# The Arena

Every solver from Phase I has a faithful Python implementation and has been run across **16 datasets** ??real and synthetic, regression and logistic ??with up to 50 hyperparameter configurations per solver where applicable.

Each *(solver, config)* run produces one coefficient vector $\widehat\beta \in \mathbb{R}^p$.  The two visualisations below make those vectors comparable:

- **Embedding map** ??unit-normalised vectors $\widehat\beta/\|\widehat\beta\|$ projected to 2D by t-SNE and UMAP.  Algorithmically similar solutions cluster; hover to see the solver name and exact config.
- **Coefficient heatmap** ??all runs for one dataset as a matrix (rows = runs, columns = features $j$). The top bar shows mean $|\widehat\beta_j|$ per feature. Colour = $\widehat\beta_j$ on a diverging RdBu scale. Hover for exact value.

All solvers share the interface `SolverCls(config={...}).fit(X, y, link=...) ??FitResult`.

---

## Real datasets

### Diabetes

<p class="arena-ds-meta">n = 442 繚 p = 10 繚 Regression 繚 UCI / sklearn</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Diabetes.html" class="arena-embed-frame" title="Embedding: Diabetes"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Diabetes.html" class="arena-heatmap-frame" title="Heatmap: Diabetes"></iframe>
</div>

### Breast Cancer

<p class="arena-ds-meta">n = 569 繚 p = 30 繚 Logistic 繚 UCI / sklearn</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Breast_Cancer.html" class="arena-embed-frame" title="Embedding: Breast Cancer"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Breast_Cancer.html" class="arena-heatmap-frame" title="Heatmap: Breast Cancer"></iframe>
</div>

### Digits (??5)

<p class="arena-ds-meta">n = 1 797 繚 p = 64 繚 Logistic 繚 NIST / sklearn</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Digits_hi5.html" class="arena-embed-frame" title="Embedding: Digits"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Digits_hi5.html" class="arena-heatmap-frame" title="Heatmap: Digits"></iframe>
</div>

### Fair Affairs

<p class="arena-ds-meta">n = 2 000 繚 p = 8 繚 Regression 繚 statsmodels</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Fair_Affairs.html" class="arena-embed-frame" title="Embedding: Fair Affairs"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Fair_Affairs.html" class="arena-heatmap-frame" title="Heatmap: Fair Affairs"></iframe>
</div>

### RAND HIE

<p class="arena-ds-meta">n = 2 000 繚 p = 9 繚 Regression 繚 statsmodels</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_RAND_HIE.html" class="arena-embed-frame" title="Embedding: RAND HIE"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_RAND_HIE.html" class="arena-heatmap-frame" title="Heatmap: RAND HIE"></iframe>
</div>

### STAR98

<p class="arena-ds-meta">n = 303 繚 p = 21 繚 Logistic 繚 statsmodels</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_STAR98_pass.html" class="arena-embed-frame" title="Embedding: STAR98"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_STAR98_pass.html" class="arena-heatmap-frame" title="Heatmap: STAR98"></iframe>
</div>

### ANES96

<p class="arena-ds-meta">n = 944 繚 p = 10 繚 Logistic 繚 statsmodels</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_ANES96_vote.html" class="arena-embed-frame" title="Embedding: ANES96"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_ANES96_vote.html" class="arena-heatmap-frame" title="Heatmap: ANES96"></iframe>
</div>

### Mode Choice

<p class="arena-ds-meta">n = 840 繚 p = 8 繚 Logistic 繚 statsmodels</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Mode_Choice.html" class="arena-embed-frame" title="Embedding: Mode Choice"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Mode_Choice.html" class="arena-heatmap-frame" title="Heatmap: Mode Choice"></iframe>
</div>

---

## Synthetic ??regression

### Sparse

<p class="arena-ds-meta">n = 2 000 繚 p = 200 繚 20 informative features 繚 Regression</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Reg_Sparse200.html" class="arena-embed-frame" title="Embedding: Sparse 200"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Reg_Sparse200.html" class="arena-heatmap-frame" title="Heatmap: Sparse 200"></iframe>
</div>

### Dense

<p class="arena-ds-meta">n = 2 000 繚 p = 100 繚 all features informative 繚 Regression</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Reg_Dense100.html" class="arena-embed-frame" title="Embedding: Dense 100"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Reg_Dense100.html" class="arena-heatmap-frame" title="Heatmap: Dense 100"></iframe>
</div>

### Correlated

<p class="arena-ds-meta">n = 2 000 繚 p = 150 繚 effective rank 10 繚 Regression</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Reg_Corr150.html" class="arena-embed-frame" title="Embedding: Correlated 150"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Reg_Corr150.html" class="arena-heatmap-frame" title="Heatmap: Correlated 150"></iframe>
</div>

### High-dimensional

<p class="arena-ds-meta">n = 2 000 繚 p = 300 繚 30 informative features 繚 Regression</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Reg_HighDim300.html" class="arena-embed-frame" title="Embedding: HighDim 300"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Reg_HighDim300.html" class="arena-heatmap-frame" title="Heatmap: HighDim 300"></iframe>
</div>

### Friedman #1

<p class="arena-ds-meta">n = 2 000 繚 p = 10 繚 nonlinear ground truth 繚 Regression</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Friedman1.html" class="arena-embed-frame" title="Embedding: Friedman1"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Friedman1.html" class="arena-heatmap-frame" title="Heatmap: Friedman1"></iframe>
</div>

---

## Synthetic ??logistic

### Dense

<p class="arena-ds-meta">n = 2 000 繚 p = 100 繚 80 informative features 繚 Logistic</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Logit_Dense100.html" class="arena-embed-frame" title="Embedding: Logit Dense100"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Logit_Dense100.html" class="arena-heatmap-frame" title="Heatmap: Logit Dense100"></iframe>
</div>

### Sparse

<p class="arena-ds-meta">n = 2 000 繚 p = 200 繚 25 informative features 繚 Logistic</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Logit_Sparse200.html" class="arena-embed-frame" title="Embedding: Logit Sparse200"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Logit_Sparse200.html" class="arena-heatmap-frame" title="Heatmap: Logit Sparse200"></iframe>
</div>

### Noisy

<p class="arena-ds-meta">n = 2 000 繚 p = 150 繚 12 % label noise 繚 Logistic</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Logit_Noisy150.html" class="arena-embed-frame" title="Embedding: Logit Noisy150"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Logit_Noisy150.html" class="arena-heatmap-frame" title="Heatmap: Logit Noisy150"></iframe>
</div>

---

## Datasets

| Dataset | Source | n | p | Task |
|---|---|---|---|---|
| Diabetes | UCI / sklearn | 442 | 10 | Regression |
| Breast Cancer | UCI / sklearn | 569 | 30 | Logistic |
| Digits (??5) | NIST / sklearn | 1 797 | 64 | Logistic |
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

