---
hide:
  - toc
---

# Nonlinear embedding maps

Each panel below shows one dataset. Every point is one *(solver, config)* run — the coefficient vector $\hat\beta$ unit-normalised to $\hat\beta / \|\hat\beta\|$, then projected to 2D by **t-SNE** (left) and **UMAP** (right).

**How to read these plots:**

- **Colour** encodes the solver. A legend appears on the right of each panel.
- **Cluster = similar solutions.** When two methods produce nearly parallel coefficient vectors their points land close together.
- **Spread within one colour = hyperparameter sensitivity.** A tight cloud means the solver's output is stable across its grid; a long streak means tuning matters.
- **Hover** over any point to see the solver name and the exact hyperparameter string.
- **Scroll to zoom**, drag to pan. Double-click to reset view.

!!! tip "Legend toggle"
    Click a solver name in the legend to hide/show that solver's points — useful when online methods crowd the plot.

---

## Real datasets

### Diabetes
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Diabetes.html" class="arena-embed-frame" title="Embedding: Diabetes"></iframe>
</div>

### Breast Cancer
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Breast_Cancer.html" class="arena-embed-frame" title="Embedding: Breast Cancer"></iframe>
</div>

### Digits (≥ 5)
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Digits_hi5.html" class="arena-embed-frame" title="Embedding: Digits"></iframe>
</div>

### Fair Affairs
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Fair_Affairs.html" class="arena-embed-frame" title="Embedding: Fair Affairs"></iframe>
</div>

### RAND HIE
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_RAND_HIE.html" class="arena-embed-frame" title="Embedding: RAND HIE"></iframe>
</div>

### STAR98 pass rate
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_STAR98_pass.html" class="arena-embed-frame" title="Embedding: STAR98"></iframe>
</div>

### ANES96 vote
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_ANES96_vote.html" class="arena-embed-frame" title="Embedding: ANES96"></iframe>
</div>

### Mode Choice
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Mode_Choice.html" class="arena-embed-frame" title="Embedding: Mode Choice"></iframe>
</div>

---

## Synthetic — regression

### Sparse (p = 200, 20 informative)
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Synth_Reg_Sparse200.html" class="arena-embed-frame" title="Embedding: Sparse 200"></iframe>
</div>

### Dense (p = 100, all informative)
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Synth_Reg_Dense100.html" class="arena-embed-frame" title="Embedding: Dense 100"></iframe>
</div>

### Correlated (p = 150, rank 10)
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Synth_Reg_Corr150.html" class="arena-embed-frame" title="Embedding: Correlated 150"></iframe>
</div>

### Very high-dimensional (p = 300, 30 informative)
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Synth_Reg_HighDim300.html" class="arena-embed-frame" title="Embedding: HighDim 300"></iframe>
</div>

### Friedman #1 (p = 10, nonlinear)
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Synth_Friedman1.html" class="arena-embed-frame" title="Embedding: Friedman1"></iframe>
</div>

---

## Synthetic — logistic

### Dense (p = 100)
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Synth_Logit_Dense100.html" class="arena-embed-frame" title="Embedding: Logit Dense100"></iframe>
</div>

### Sparse (p = 200)
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Synth_Logit_Sparse200.html" class="arena-embed-frame" title="Embedding: Logit Sparse200"></iframe>
</div>

### Noisy (p = 150, 12% label noise)
<div class="arena-figure-frame">
<iframe src="../figures/tsne_umap_Synth_Logit_Noisy150.html" class="arena-embed-frame" title="Embedding: Logit Noisy150"></iframe>
</div>
