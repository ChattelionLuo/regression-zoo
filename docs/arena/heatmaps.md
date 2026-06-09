---
hide:
  - toc
---

# Coefficient heatmaps

Each panel shows all *(solver, config)* runs for one dataset as a matrix:

- **Rows** — one row per successful run, sorted by solver family → solver name → $\|\hat\beta\|$.
- **Columns** — feature dimensions $j = 0 \ldots p-1$.
- **Colour** — diverging RdBu scale centred at 0; symmetric at the 99th percentile of $|\hat\beta_j|$.

Solvers with a single configuration (OLS, GLM-IRLS) are displayed as a widened band so they remain visible alongside the 50-config sweeps.

**How to read:**

- A **solid-colour horizontal band** means the solver uses all features roughly equally (Ridge, dense methods).
- A **sparse band** (mostly white/near-zero) means the solver zeroes most features (Lasso, SCAD, MCP at high regularization).
- **Structured vertical patterns** (Group Lasso, Fused Lasso) reveal group or smoothness constraints acting on adjacent features.
- **Hover** over any cell to read the exact $\hat\beta_j$ value, solver name, and config string.
- **Scroll to zoom**, drag to pan.

---

## Real datasets

### Diabetes
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Diabetes.html" class="arena-heatmap-frame" title="Heatmap: Diabetes"></iframe>
</div>

### Breast Cancer
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Breast_Cancer.html" class="arena-heatmap-frame" title="Heatmap: Breast Cancer"></iframe>
</div>

### Digits (≥ 5)
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Digits_hi5.html" class="arena-heatmap-frame" title="Heatmap: Digits"></iframe>
</div>

### Fair Affairs
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Fair_Affairs.html" class="arena-heatmap-frame" title="Heatmap: Fair Affairs"></iframe>
</div>

### RAND HIE
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_RAND_HIE.html" class="arena-heatmap-frame" title="Heatmap: RAND HIE"></iframe>
</div>

### STAR98 pass rate
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_STAR98_pass.html" class="arena-heatmap-frame" title="Heatmap: STAR98"></iframe>
</div>

### ANES96 vote
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_ANES96_vote.html" class="arena-heatmap-frame" title="Heatmap: ANES96"></iframe>
</div>

### Mode Choice
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Mode_Choice.html" class="arena-heatmap-frame" title="Heatmap: Mode Choice"></iframe>
</div>

---

## Synthetic — regression

### Sparse (p = 200, 20 informative)
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Synth_Reg_Sparse200.html" class="arena-heatmap-frame" title="Heatmap: Sparse 200"></iframe>
</div>

### Dense (p = 100, all informative)
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Synth_Reg_Dense100.html" class="arena-heatmap-frame" title="Heatmap: Dense 100"></iframe>
</div>

### Correlated (p = 150, rank 10)
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Synth_Reg_Corr150.html" class="arena-heatmap-frame" title="Heatmap: Correlated 150"></iframe>
</div>

### Very high-dimensional (p = 300, 30 informative)
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Synth_Reg_HighDim300.html" class="arena-heatmap-frame" title="Heatmap: HighDim 300"></iframe>
</div>

### Friedman #1 (p = 10, nonlinear)
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Synth_Friedman1.html" class="arena-heatmap-frame" title="Heatmap: Friedman1"></iframe>
</div>

---

## Synthetic — logistic

### Dense (p = 100)
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Synth_Logit_Dense100.html" class="arena-heatmap-frame" title="Heatmap: Logit Dense100"></iframe>
</div>

### Sparse (p = 200)
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Synth_Logit_Sparse200.html" class="arena-heatmap-frame" title="Heatmap: Logit Sparse200"></iframe>
</div>

### Noisy (p = 150, 12% label noise)
<div class="arena-figure-frame">
<iframe src="../figures/heatmap_Synth_Logit_Noisy150.html" class="arena-heatmap-frame" title="Heatmap: Logit Noisy150"></iframe>
</div>
