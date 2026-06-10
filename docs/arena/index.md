# The Arena

Every [algorithm](algorithms/index.md) has a Python implementation and has been run across all datasets, real and synthetic, regression and logistic, with up to 50 hyperparameter configurations per algorithm where applicable.

Each (`algorithm`, `config`) run produces one coefficient vector $\widehat\beta \in \mathbb{R}^p$.  At this time we have two visualisations:

- **Embedding map**: unit-normalised vectors $\widehat\beta/\|\widehat\beta\|$ projected to 2D by t-SNE and UMAP.  Algorithmically similar solutions cluster; hover to see the algorithm name and exact config.
- **Coefficient heatmap**: all runs for one dataset as a matrix (rows = runs, columns = feature indices). The top bar shows mean $|\widehat\beta_j|$ per feature. Hover for exact value.

All algorithms share the interface `SolverCls(config={...}).fit(X, y, link=...) -> FitResult`.

---

!!! tip "Tip"
    This page is best explored on a desktop or laptop browser, where you can zoom, pan, and hover. If you want to suggest a dataset or a new arena function, feel free to reach out at <a href="mailto:chattelion.luo@connect.polyu.hk">chattelion.luo@connect.polyu.hk</a>

## Real datasets

### Diabetes

<p class="arena-ds-meta">n = 442 · p = 10 · Regression · UCI / sklearn</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Diabetes.html" class="arena-embed-frame" title="Embedding: Diabetes"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Diabetes.html" class="arena-heatmap-frame" title="Heatmap: Diabetes"></iframe>
</div>

### Breast Cancer

<p class="arena-ds-meta">n = 569 · p = 30 · Logistic · UCI / sklearn</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Breast_Cancer.html" class="arena-embed-frame" title="Embedding: Breast Cancer"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Breast_Cancer.html" class="arena-heatmap-frame" title="Heatmap: Breast Cancer"></iframe>
</div>

### Digits (>= 5)

<p class="arena-ds-meta">n = 1 797 · p = 64 · Logistic · NIST / sklearn</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Digits_hi5.html" class="arena-embed-frame" title="Embedding: Digits"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Digits_hi5.html" class="arena-heatmap-frame" title="Heatmap: Digits"></iframe>
</div>

### Fair Affairs

<p class="arena-ds-meta">n = 2 000 · p = 8 · Regression · statsmodels</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Fair_Affairs.html" class="arena-embed-frame" title="Embedding: Fair Affairs"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Fair_Affairs.html" class="arena-heatmap-frame" title="Heatmap: Fair Affairs"></iframe>
</div>

### RAND HIE

<p class="arena-ds-meta">n = 2 000 · p = 9 · Regression · statsmodels</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_RAND_HIE.html" class="arena-embed-frame" title="Embedding: RAND HIE"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_RAND_HIE.html" class="arena-heatmap-frame" title="Heatmap: RAND HIE"></iframe>
</div>

### STAR98

<p class="arena-ds-meta">n = 303 · p = 21 · Logistic · statsmodels</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_STAR98_pass.html" class="arena-embed-frame" title="Embedding: STAR98"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_STAR98_pass.html" class="arena-heatmap-frame" title="Heatmap: STAR98"></iframe>
</div>

### ANES96

<p class="arena-ds-meta">n = 944 · p = 10 · Logistic · statsmodels</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_ANES96_vote.html" class="arena-embed-frame" title="Embedding: ANES96"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_ANES96_vote.html" class="arena-heatmap-frame" title="Heatmap: ANES96"></iframe>
</div>

### Mode Choice

<p class="arena-ds-meta">n = 840 · p = 8 · Logistic · statsmodels</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Mode_Choice.html" class="arena-embed-frame" title="Embedding: Mode Choice"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Mode_Choice.html" class="arena-heatmap-frame" title="Heatmap: Mode Choice"></iframe>
</div>

---

## Synthetic — regression

### Sparse

<p class="arena-ds-meta">n = 2 000 · p = 200 · 20 informative features · Regression</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Reg_Sparse200.html" class="arena-embed-frame" title="Embedding: Sparse 200"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Reg_Sparse200.html" class="arena-heatmap-frame" title="Heatmap: Sparse 200"></iframe>
</div>

### Dense

<p class="arena-ds-meta">n = 2 000 · p = 100 · all features informative · Regression</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Reg_Dense100.html" class="arena-embed-frame" title="Embedding: Dense 100"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Reg_Dense100.html" class="arena-heatmap-frame" title="Heatmap: Dense 100"></iframe>
</div>

### Correlated

<p class="arena-ds-meta">n = 2 000 · p = 150 · effective rank 10 · Regression</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Reg_Corr150.html" class="arena-embed-frame" title="Embedding: Correlated 150"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Reg_Corr150.html" class="arena-heatmap-frame" title="Heatmap: Correlated 150"></iframe>
</div>

### High-dimensional

<p class="arena-ds-meta">n = 2 000 · p = 300 · 30 informative features · Regression</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Reg_HighDim300.html" class="arena-embed-frame" title="Embedding: HighDim 300"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Reg_HighDim300.html" class="arena-heatmap-frame" title="Heatmap: HighDim 300"></iframe>
</div>

### Friedman #1

<p class="arena-ds-meta">n = 2 000 · p = 10 · nonlinear ground truth · Regression</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Friedman1.html" class="arena-embed-frame" title="Embedding: Friedman1"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Friedman1.html" class="arena-heatmap-frame" title="Heatmap: Friedman1"></iframe>
</div>

---

## Synthetic — logistic

### Dense

<p class="arena-ds-meta">n = 2 000 · p = 100 · 80 informative features · Logistic</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Logit_Dense100.html" class="arena-embed-frame" title="Embedding: Logit Dense100"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Logit_Dense100.html" class="arena-heatmap-frame" title="Heatmap: Logit Dense100"></iframe>
</div>

### Sparse

<p class="arena-ds-meta">n = 2 000 · p = 200 · 25 informative features · Logistic</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_Synth_Logit_Sparse200.html" class="arena-embed-frame" title="Embedding: Logit Sparse200"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_Synth_Logit_Sparse200.html" class="arena-heatmap-frame" title="Heatmap: Logit Sparse200"></iframe>
</div>

### Noisy

<p class="arena-ds-meta">n = 2 000 · p = 150 · 12 % label noise · Logistic</p>

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



## LIBSVM Benchmark Datasets

The following visualisations cover **38 datasets** from the [LIBSVM data repository](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/), comprising **3 606 solver runs** across regression and binary-classification tasks. Datasets span from tiny (n = 38, p = 7 129 for Leukemia) to large (n = 80 000, p = 2 000 for Epsilon).

### Regression

### Abalone

<p class="arena-ds-meta">n = 4,177 &middot; p = 8 &middot; Regression &middot; LIBSVM &middot; 100 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_abalone.html" class="arena-embed-frame" title="Embedding: Abalone"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_abalone.html" class="arena-heatmap-frame" title="Heatmap: Abalone"></iframe>
</div>

### Body Fat

<p class="arena-ds-meta">n = 252 &middot; p = 14 &middot; Regression &middot; LIBSVM &middot; 101 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_bodyfat.html" class="arena-embed-frame" title="Embedding: Body Fat"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_bodyfat.html" class="arena-heatmap-frame" title="Heatmap: Body Fat"></iframe>
</div>

### California Housing (cadata)

<p class="arena-ds-meta">n = 20,640 &middot; p = 8 &middot; Regression &middot; LIBSVM &middot; 100 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_cadata.html" class="arena-embed-frame" title="Embedding: California Housing (cadata)"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_cadata.html" class="arena-heatmap-frame" title="Heatmap: California Housing (cadata)"></iframe>
</div>

### CPU Small

<p class="arena-ds-meta">n = 8,192 &middot; p = 12 &middot; Regression &middot; LIBSVM &middot; 101 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_cpusmall.html" class="arena-embed-frame" title="Embedding: CPU Small"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_cpusmall.html" class="arena-heatmap-frame" title="Heatmap: CPU Small"></iframe>
</div>

### EUNITE 2001 Electricity

<p class="arena-ds-meta">n = 336 &middot; p = 16 &middot; Regression &middot; LIBSVM &middot; 101 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_eunite2001.html" class="arena-embed-frame" title="Embedding: EUNITE 2001 Electricity"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_eunite2001.html" class="arena-heatmap-frame" title="Heatmap: EUNITE 2001 Electricity"></iframe>
</div>

### Boston Housing

<p class="arena-ds-meta">n = 506 &middot; p = 13 &middot; Regression &middot; LIBSVM &middot; 101 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_housing.html" class="arena-embed-frame" title="Embedding: Boston Housing"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_housing.html" class="arena-heatmap-frame" title="Heatmap: Boston Housing"></iframe>
</div>

### Mackey-Glass (mg)

<p class="arena-ds-meta">n = 1,385 &middot; p = 6 &middot; Regression &middot; LIBSVM &middot; 100 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_mg.html" class="arena-embed-frame" title="Embedding: Mackey-Glass (mg)"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_mg.html" class="arena-heatmap-frame" title="Heatmap: Mackey-Glass (mg)"></iframe>
</div>

### Auto MPG

<p class="arena-ds-meta">n = 392 &middot; p = 7 &middot; Regression &middot; LIBSVM &middot; 100 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_mpg.html" class="arena-embed-frame" title="Embedding: Auto MPG"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_mpg.html" class="arena-heatmap-frame" title="Heatmap: Auto MPG"></iframe>
</div>

### Pyrimidines

<p class="arena-ds-meta">n = 74 &middot; p = 27 &middot; Regression &middot; LIBSVM &middot; 102 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_pyrim.html" class="arena-embed-frame" title="Embedding: Pyrimidines"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_pyrim.html" class="arena-heatmap-frame" title="Heatmap: Pyrimidines"></iframe>
</div>

### Space GA

<p class="arena-ds-meta">n = 3,107 &middot; p = 6 &middot; Regression &middot; LIBSVM &middot; 100 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_space_ga.html" class="arena-embed-frame" title="Embedding: Space GA"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_space_ga.html" class="arena-heatmap-frame" title="Heatmap: Space GA"></iframe>
</div>

### Triazines

<p class="arena-ds-meta">n = 186 &middot; p = 60 &middot; Regression &middot; LIBSVM &middot; 103 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_triazines.html" class="arena-embed-frame" title="Embedding: Triazines"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_triazines.html" class="arena-heatmap-frame" title="Heatmap: Triazines"></iframe>
</div>

### Year Prediction MSD

<p class="arena-ds-meta">n = 80,000 &middot; p = 90 &middot; Regression &middot; LIBSVM &middot; 103 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_YearPredMSD.html" class="arena-embed-frame" title="Embedding: Year Prediction MSD"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_YearPredMSD.html" class="arena-heatmap-frame" title="Heatmap: Year Prediction MSD"></iframe>
</div>

### Binary Classification

### Australian Credit

<p class="arena-ds-meta">n = 690 &middot; p = 14 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_australian.html" class="arena-embed-frame" title="Embedding: Australian Credit"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_australian.html" class="arena-heatmap-frame" title="Heatmap: Australian Credit"></iframe>
</div>

### Breast Cancer (LIBSVM)

<p class="arena-ds-meta">n = 683 &middot; p = 10 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_breast-cancer.html" class="arena-embed-frame" title="Embedding: Breast Cancer (LIBSVM)"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_breast-cancer.html" class="arena-heatmap-frame" title="Heatmap: Breast Cancer (LIBSVM)"></iframe>
</div>

### COD-RNA

<p class="arena-ds-meta">n = 59,535 &middot; p = 8 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_cod-rna.html" class="arena-embed-frame" title="Embedding: COD-RNA"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_cod-rna.html" class="arena-heatmap-frame" title="Heatmap: COD-RNA"></iframe>
</div>

### Colon Cancer

<p class="arena-ds-meta">n = 62 &middot; p = 2,000 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_colon-cancer.html" class="arena-embed-frame" title="Embedding: Colon Cancer"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_colon-cancer.html" class="arena-heatmap-frame" title="Heatmap: Colon Cancer"></iframe>
</div>

### Diabetes (Pima)

<p class="arena-ds-meta">n = 768 &middot; p = 8 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_diabetes.html" class="arena-embed-frame" title="Embedding: Diabetes (Pima)"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_diabetes.html" class="arena-heatmap-frame" title="Heatmap: Diabetes (Pima)"></iframe>
</div>

### Duke Breast Cancer

<p class="arena-ds-meta">n = 44 &middot; p = 7,129 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_duke.html" class="arena-embed-frame" title="Embedding: Duke Breast Cancer"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_duke.html" class="arena-heatmap-frame" title="Heatmap: Duke Breast Cancer"></iframe>
</div>

### Epsilon

<p class="arena-ds-meta">n = 80,000 &middot; p = 2,000 &middot; Binary logistic &middot; LIBSVM &middot; 56 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_epsilon.html" class="arena-embed-frame" title="Embedding: Epsilon"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_epsilon.html" class="arena-heatmap-frame" title="Heatmap: Epsilon"></iframe>
</div>

### Fourclass

<p class="arena-ds-meta">n = 862 &middot; p = 2 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_fourclass.html" class="arena-embed-frame" title="Embedding: Fourclass"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_fourclass.html" class="arena-heatmap-frame" title="Heatmap: Fourclass"></iframe>
</div>

### German Credit (numerical)

<p class="arena-ds-meta">n = 1,000 &middot; p = 24 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_german-numer.html" class="arena-embed-frame" title="Embedding: German Credit (numerical)"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_german-numer.html" class="arena-heatmap-frame" title="Heatmap: German Credit (numerical)"></iframe>
</div>

### Gisette

<p class="arena-ds-meta">n = 6,000 &middot; p = 5,000 &middot; Binary logistic &middot; LIBSVM &middot; 43 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_gisette.html" class="arena-embed-frame" title="Embedding: Gisette"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_gisette.html" class="arena-heatmap-frame" title="Heatmap: Gisette"></iframe>
</div>

### Heart Disease

<p class="arena-ds-meta">n = 270 &middot; p = 13 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_heart.html" class="arena-embed-frame" title="Embedding: Heart Disease"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_heart.html" class="arena-heatmap-frame" title="Heatmap: Heart Disease"></iframe>
</div>

### IJCNN1

<p class="arena-ds-meta">n = 49,990 &middot; p = 22 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_ijcnn1.html" class="arena-embed-frame" title="Embedding: IJCNN1"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_ijcnn1.html" class="arena-heatmap-frame" title="Heatmap: IJCNN1"></iframe>
</div>

### Ionosphere

<p class="arena-ds-meta">n = 351 &middot; p = 34 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_ionosphere.html" class="arena-embed-frame" title="Embedding: Ionosphere"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_ionosphere.html" class="arena-heatmap-frame" title="Heatmap: Ionosphere"></iframe>
</div>

### Leukemia

<p class="arena-ds-meta">n = 38 &middot; p = 7,129 &middot; Binary logistic &middot; LIBSVM &middot; 85 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_leukemia.html" class="arena-embed-frame" title="Embedding: Leukemia"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_leukemia.html" class="arena-heatmap-frame" title="Heatmap: Leukemia"></iframe>
</div>

### Liver Disorders

<p class="arena-ds-meta">n = 145 &middot; p = 5 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_liver-disorders.html" class="arena-embed-frame" title="Embedding: Liver Disorders"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_liver-disorders.html" class="arena-heatmap-frame" title="Heatmap: Liver Disorders"></iframe>
</div>

### Madelon

<p class="arena-ds-meta">n = 2,000 &middot; p = 500 &middot; Binary logistic &middot; LIBSVM &middot; 97 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_madelon.html" class="arena-embed-frame" title="Embedding: Madelon"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_madelon.html" class="arena-heatmap-frame" title="Heatmap: Madelon"></iframe>
</div>

### Mushrooms

<p class="arena-ds-meta">n = 8,124 &middot; p = 112 &middot; Binary logistic &middot; LIBSVM &middot; 97 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_mushrooms.html" class="arena-embed-frame" title="Embedding: Mushrooms"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_mushrooms.html" class="arena-heatmap-frame" title="Heatmap: Mushrooms"></iframe>
</div>

### Phishing Websites

<p class="arena-ds-meta">n = 11,055 &middot; p = 68 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_phishing.html" class="arena-embed-frame" title="Embedding: Phishing Websites"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_phishing.html" class="arena-heatmap-frame" title="Heatmap: Phishing Websites"></iframe>
</div>

### Skin/Non-Skin

<p class="arena-ds-meta">n = 80,000 &middot; p = 3 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_skin-nonskin.html" class="arena-embed-frame" title="Embedding: Skin/Non-Skin"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_skin-nonskin.html" class="arena-heatmap-frame" title="Heatmap: Skin/Non-Skin"></iframe>
</div>

### Sonar

<p class="arena-ds-meta">n = 208 &middot; p = 60 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_sonar.html" class="arena-embed-frame" title="Embedding: Sonar"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_sonar.html" class="arena-heatmap-frame" title="Heatmap: Sonar"></iframe>
</div>

### Splice

<p class="arena-ds-meta">n = 1,000 &middot; p = 60 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_splice.html" class="arena-embed-frame" title="Embedding: Splice"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_splice.html" class="arena-heatmap-frame" title="Heatmap: Splice"></iframe>
</div>

### SVMguide1

<p class="arena-ds-meta">n = 3,089 &middot; p = 4 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_svmguide1.html" class="arena-embed-frame" title="Embedding: SVMguide1"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_svmguide1.html" class="arena-heatmap-frame" title="Heatmap: SVMguide1"></iframe>
</div>

### SVMguide3

<p class="arena-ds-meta">n = 1,243 &middot; p = 22 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_svmguide3.html" class="arena-embed-frame" title="Embedding: SVMguide3"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_svmguide3.html" class="arena-heatmap-frame" title="Heatmap: SVMguide3"></iframe>
</div>

### w8a

<p class="arena-ds-meta">n = 49,749 &middot; p = 300 &middot; Binary logistic &middot; LIBSVM &middot; 56 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_w8a.html" class="arena-embed-frame" title="Embedding: w8a"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_w8a.html" class="arena-heatmap-frame" title="Heatmap: w8a"></iframe>
</div>

### a9a

<p class="arena-ds-meta">n = 32,561 &middot; p = 123 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_a9a.html" class="arena-embed-frame" title="Embedding: a9a"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_a9a.html" class="arena-heatmap-frame" title="Heatmap: a9a"></iframe>
</div>

### Covertype (binary)

<p class="arena-ds-meta">n = 80,000 &middot; p = 54 &middot; Binary logistic &middot; LIBSVM &middot; 98 runs</p>

<div class="arena-figure-frame">
<iframe src="figures/tsne_umap_libsvm_covtype-binary.html" class="arena-embed-frame" title="Embedding: Covertype (binary)"></iframe>
</div>
<div class="arena-figure-frame">
<iframe src="figures/heatmap_libsvm_covtype-binary.html" class="arena-heatmap-frame" title="Heatmap: Covertype (binary)"></iframe>
</div>

## Dataset Summary

| Dataset | n | p | Task | Runs |
|---------|---|---|------|------|
| Abalone | 4,177 | 8 | Regression | 100 |
| Body Fat | 252 | 14 | Regression | 101 |
| California Housing (cadata) | 20,640 | 8 | Regression | 100 |
| CPU Small | 8,192 | 12 | Regression | 101 |
| EUNITE 2001 Electricity | 336 | 16 | Regression | 101 |
| Boston Housing | 506 | 13 | Regression | 101 |
| Mackey-Glass (mg) | 1,385 | 6 | Regression | 100 |
| Auto MPG | 392 | 7 | Regression | 100 |
| Pyrimidines | 74 | 27 | Regression | 102 |
| Space GA | 3,107 | 6 | Regression | 100 |
| Triazines | 186 | 60 | Regression | 103 |
| Year Prediction MSD | 80,000 | 90 | Regression | 103 |
| Australian Credit | 690 | 14 | Binary logistic | 98 |
| Breast Cancer (LIBSVM) | 683 | 10 | Binary logistic | 98 |
| COD-RNA | 59,535 | 8 | Binary logistic | 98 |
| Colon Cancer | 62 | 2,000 | Binary logistic | 98 |
| Diabetes (Pima) | 768 | 8 | Binary logistic | 98 |
| Duke Breast Cancer | 44 | 7,129 | Binary logistic | 98 |
| Epsilon | 80,000 | 2,000 | Binary logistic | 56 |
| Fourclass | 862 | 2 | Binary logistic | 98 |
| German Credit (numerical) | 1,000 | 24 | Binary logistic | 98 |
| Gisette | 6,000 | 5,000 | Binary logistic | 43 |
| Heart Disease | 270 | 13 | Binary logistic | 98 |
| IJCNN1 | 49,990 | 22 | Binary logistic | 98 |
| Ionosphere | 351 | 34 | Binary logistic | 98 |
| Leukemia | 38 | 7,129 | Binary logistic | 85 |
| Liver Disorders | 145 | 5 | Binary logistic | 98 |
| Madelon | 2,000 | 500 | Binary logistic | 97 |
| Mushrooms | 8,124 | 112 | Binary logistic | 97 |
| Phishing Websites | 11,055 | 68 | Binary logistic | 98 |
| Skin/Non-Skin | 80,000 | 3 | Binary logistic | 98 |
| Sonar | 208 | 60 | Binary logistic | 98 |
| Splice | 1,000 | 60 | Binary logistic | 98 |
| SVMguide1 | 3,089 | 4 | Binary logistic | 98 |
| SVMguide3 | 1,243 | 22 | Binary logistic | 98 |
| w8a | 49,749 | 300 | Binary logistic | 56 |
| a9a | 32,561 | 123 | Binary logistic | 98 |
| Covertype (binary) | 80,000 | 54 | Binary logistic | 98 |

