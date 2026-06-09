"""
generate_arena_plots.py
=======================
Generates interactive Plotly HTML figures for the Arena website section.

Outputs (written to docs/arena/figures/):
  - tsne_umap_<dataset>.html   — per-dataset interactive t-SNE + UMAP scatter
  - heatmap_<dataset>.html     — per-dataset interactive coefficient heatmap

Also writes docs/arena/figures/datasets.json for use in the overview page.

Usage:
  python scripts/generate_arena_plots.py [--no-cache]

Cache:
  notebooks/arena_results.pkl  — saved after first run; --no-cache forces re-run.
"""

from __future__ import annotations
import sys, os, time, warnings, itertools, json, pickle, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.datasets import load_diabetes, load_breast_cancer, load_digits
from sklearn.datasets import make_regression, make_classification, make_friedman1
import sklearn
import umap as umap_lib

# ── Output directory ────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'arena', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'arena_results.pkl')

# ── Solver imports ───────────────────────────────────────────────────────────
from src.glmzoo.solvers.classical.ols             import OLSSolver
from src.glmzoo.solvers.classical.ridge           import RidgeSolver
from src.glmzoo.solvers.classical.glm_irls        import GLMIRLSSolver
from src.glmzoo.solvers.penalized.lasso_cd        import LassoCDSolver
from src.glmzoo.solvers.penalized.elastic_net     import ElasticNetSolver
from src.glmzoo.solvers.penalized.adaptive_lasso  import AdaptiveLassoSolver
from src.glmzoo.solvers.penalized.scad_lla        import SCADLLASolver
from src.glmzoo.solvers.penalized.mcp_cd          import MCPCDSolver
from src.glmzoo.solvers.penalized.group_lasso     import GroupLassoSolver
from src.glmzoo.solvers.penalized.fused_lasso     import FusedLassoSolver
from src.glmzoo.solvers.path.lars                 import LARSSolver
from src.glmzoo.solvers.first_order.ista          import ISTASolver
from src.glmzoo.solvers.first_order.fista         import FISTASolver
from src.glmzoo.solvers.online.sgd                import SGDSolver
from src.glmzoo.solvers.online.implicit_sgd       import ImplicitSGDSolver
from src.glmzoo.solvers.online.adagrad            import AdaGradSolver
from src.glmzoo.solvers.online.fobos              import FOBOSSolver
from src.glmzoo.solvers.online.rda                import RDASolver
from src.glmzoo.solvers.online.truncated_gradient import TruncatedGradientSolver
from src.glmzoo.solvers.online.renewable_glm      import RenewableGLMSolver

# ── Metadata ─────────────────────────────────────────────────────────────────
FAMILY = {
    OLSSolver: "Classical", GLMIRLSSolver: "Classical", LARSSolver: "Classical",
    RidgeSolver: "Penalized", LassoCDSolver: "Penalized", ElasticNetSolver: "Penalized",
    AdaptiveLassoSolver: "Penalized", SCADLLASolver: "Penalized",
    MCPCDSolver: "Penalized", GroupLassoSolver: "Penalized", FusedLassoSolver: "Penalized",
    ISTASolver: "First-order", FISTASolver: "First-order",
    SGDSolver: "Online", ImplicitSGDSolver: "Online", AdaGradSolver: "Online",
    FOBOSSolver: "Online", RDASolver: "Online", TruncatedGradientSolver: "Online",
    RenewableGLMSolver: "Online",
}

ALGO_LABELS = {
    "OLSSolver": "OLS", "GLMIRLSSolver": "GLM-IRLS", "LARSSolver": "LARS",
    "RidgeSolver": "Ridge", "LassoCDSolver": "Lasso-CD",
    "ElasticNetSolver": "Elastic Net", "AdaptiveLassoSolver": "Adaptive Lasso",
    "SCADLLASolver": "SCAD", "MCPCDSolver": "MCP-CD",
    "GroupLassoSolver": "Group Lasso", "FusedLassoSolver": "Fused Lasso",
    "ISTASolver": "ISTA", "FISTASolver": "FISTA",
    "SGDSolver": "SGD", "ImplicitSGDSolver": "Implicit SGD",
    "AdaGradSolver": "AdaGrad", "FOBOSSolver": "FOBOS",
    "RDASolver": "RDA", "TruncatedGradientSolver": "Trunc. Gradient",
    "RenewableGLMSolver": "Renewable GLM",
}

# Consistent colour per solver (20-colour qualitative palette)
_SOLVER_ORDER = [
    OLSSolver, GLMIRLSSolver, LARSSolver,
    RidgeSolver, LassoCDSolver, ElasticNetSolver,
    AdaptiveLassoSolver, SCADLLASolver, MCPCDSolver,
    GroupLassoSolver, FusedLassoSolver,
    ISTASolver, FISTASolver,
    SGDSolver, ImplicitSGDSolver, AdaGradSolver,
    FOBOSSolver, RDASolver, TruncatedGradientSolver, RenewableGLMSolver,
]

# Use a professional colour palette — matching site's near-black + accent blue
_PALETTE = [
    "#1a1a2e", "#16213e", "#0f3460", "#533483",
    "#2d6a4f", "#40916c", "#74c69d", "#b7e4c7",
    "#7b2d8b", "#b5179e", "#f72585", "#ff6b6b",
    "#ee9b00", "#ca6702", "#bb3e03", "#ae2012",
    "#005f73", "#0a9396", "#94d2bd", "#e9d8a6",
]
ALGO_COLORS = {cls.__name__: _PALETTE[i] for i, cls in enumerate(_SOLVER_ORDER)}

SKIP_FOR_LOGISTIC = {LARSSolver, FusedLassoSolver}

# ── Dataset loading ───────────────────────────────────────────────────────────
def _add_regression(datasets, X, y, name, MAX_N=2000):
    rng = np.random.default_rng(42)
    X = np.array(X, dtype=float); y = np.array(y, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(X) < 50: return
    if len(X) > MAX_N:
        idx = rng.choice(len(X), MAX_N, replace=False); X, y = X[idx], y[idx]
    X = StandardScaler().fit_transform(X)
    y = (y - y.mean()) / (y.std() + 1e-12)
    datasets[name] = dict(X=X, y=y, link='identity', kind='regression')

def _add_logistic(datasets, X, y, name, MAX_N=2000):
    rng = np.random.default_rng(42)
    X = np.array(X, dtype=float); y = np.array(y, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(X) < 50: return
    if len(X) > MAX_N:
        idx = rng.choice(len(X), MAX_N, replace=False); X, y = X[idx], y[idx]
    X = StandardScaler().fit_transform(X)
    datasets[name] = dict(X=X, y=y, link='logit', kind='logistic')

def load_datasets():
    datasets = {}
    rng = np.random.default_rng(42)

    d = load_diabetes(); _add_regression(datasets, d.data, d.target, 'Diabetes')
    d = load_breast_cancer(); _add_logistic(datasets, d.data, d.target.astype(float), 'Breast_Cancer')
    d = load_digits(); _add_logistic(datasets, d.data, (d.target >= 5).astype(float), 'Digits_hi5')

    try:
        import statsmodels.datasets as smd
        d = smd.fair.load_pandas().data
        _add_regression(datasets, d.drop(columns=['affairs']).select_dtypes(include=[float,'int64']).values, d['affairs'].values, 'Fair_Affairs')
        d = smd.randhie.load_pandas().data.select_dtypes(include=[float,'int64']).dropna()
        yc = 'lnmeddol' if 'lnmeddol' in d.columns else d.columns[0]
        _add_regression(datasets, d.drop(columns=[yc]).values, d[yc].values, 'RAND_HIE')
        d = smd.star98.load_pandas().data.select_dtypes(include=[float,'int64']).dropna()
        y0 = d.iloc[:,0].values; X0 = d.iloc[:,1:].values
        _add_logistic(datasets, X0, (y0 > np.median(y0)).astype(float), 'STAR98_pass')
        d = smd.anes96.load_pandas().data.select_dtypes(include=[float,'int64']).dropna()
        y0 = d.iloc[:,0].values; X0 = d.iloc[:,1:].values
        _add_logistic(datasets, X0, (y0 > np.median(y0)).astype(float), 'ANES96_vote')
        d = smd.modechoice.load_pandas().data.select_dtypes(include=[float,'int64']).dropna()
        y0 = d.iloc[:,0].values; X0 = d.iloc[:,1:].values
        _add_logistic(datasets, X0, (y0 > np.median(y0)).astype(float), 'Mode_Choice')
    except Exception as e:
        print(f"  statsmodels error: {e}")

    Xs, ys = make_regression(n_samples=2000, n_features=200, n_informative=20, noise=0.5, random_state=1); _add_regression(datasets, Xs, ys, 'Synth_Reg_Sparse200')
    Xs, ys = make_regression(n_samples=2000, n_features=100, n_informative=100, noise=0.3, random_state=2); _add_regression(datasets, Xs, ys, 'Synth_Reg_Dense100')
    Xs, ys = make_regression(n_samples=2000, n_features=150, n_informative=80, effective_rank=10, noise=0.4, random_state=3); _add_regression(datasets, Xs, ys, 'Synth_Reg_Corr150')
    Xs, ys = make_regression(n_samples=2000, n_features=300, n_informative=30, noise=1.0, random_state=4); _add_regression(datasets, Xs, ys, 'Synth_Reg_HighDim300')
    Xs, ys = make_friedman1(n_samples=2000, n_features=10, noise=0.5, random_state=42); _add_regression(datasets, Xs, ys, 'Synth_Friedman1')
    Xs, ys = make_classification(n_samples=2000, n_features=100, n_informative=80, n_redundant=15, random_state=5); _add_logistic(datasets, Xs, ys.astype(float), 'Synth_Logit_Dense100')
    Xs, ys = make_classification(n_samples=2000, n_features=200, n_informative=25, n_redundant=50, random_state=6); _add_logistic(datasets, Xs, ys.astype(float), 'Synth_Logit_Sparse200')
    Xs, ys = make_classification(n_samples=2000, n_features=150, n_informative=40, n_redundant=40, flip_y=0.12, random_state=7); _add_logistic(datasets, Xs, ys.astype(float), 'Synth_Logit_Noisy150')

    return datasets

# ── Config builder ─────────────────────────────────────────────────────────────
def compute_lam_max(X, y):
    return float(np.max(np.abs(X.T @ (y - y.mean()))) / len(y))

def make_groups(p, n_groups):
    n_groups = max(1, min(n_groups, p))
    return [list(range(j, p, n_groups)) for j in range(n_groups)]

def build_configs(lam_max, p, kind='regression'):
    cfgs = []
    def lam_grid(n=50, frac_min=0.001):
        lo = max(lam_max * frac_min, 1e-7)
        hi = max(lam_max, lo * 2)
        return np.logspace(np.log10(lo), np.log10(hi), n)

    cfgs.append((OLSSolver,     {}, 'OLS'))
    cfgs.append((GLMIRLSSolver, {}, 'IRLS'))

    if kind == 'regression':
        for lam in lam_grid(50): cfgs.append((LARSSolver, {'lam': float(lam)}, f'LARS λ={lam:.4g}'))

    for lam in lam_grid(50): cfgs.append((RidgeSolver,   {'lam': float(lam)}, f'Ridge λ={lam:.4g}'))
    for lam in lam_grid(50): cfgs.append((LassoCDSolver, {'lam': float(lam)}, f'Lasso λ={lam:.4g}'))
    for lam in lam_grid(50): cfgs.append((ISTASolver,    {'lam': float(lam)}, f'ISTA λ={lam:.4g}'))
    for lam in lam_grid(50): cfgs.append((FISTASolver,   {'lam': float(lam)}, f'FISTA λ={lam:.4g}'))

    for lam in lam_grid(10):
        for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
            cfgs.append((ElasticNetSolver, {'lam': float(lam), 'alpha': a}, f'EN λ={lam:.4g} α={a}'))
    for lam in lam_grid(10):
        for g in [0.5, 1.0, 2.0, 3.0, 5.0]:
            cfgs.append((AdaptiveLassoSolver, {'lam': float(lam), 'gamma': g}, f'AdaLasso λ={lam:.4g} γ={g}'))
    for lam in lam_grid(10):
        for a in [2.5, 3.7, 5.0, 7.0, 10.0]:
            cfgs.append((SCADLLASolver, {'lam': float(lam), 'a': a}, f'SCAD λ={lam:.4g} a={a}'))
    for lam in lam_grid(10):
        for g in [1.5, 2.0, 3.0, 5.0, 8.0]:
            cfgs.append((MCPCDSolver, {'lam': float(lam), 'gamma': g}, f'MCP λ={lam:.4g} γ={g}'))

    ng_vals = sorted(set([2, max(2, p//10), max(2, p//5), max(2, p//3), min(p, max(5, p//2))]))[:5]
    for lam in lam_grid(10):
        for ng in ng_vals:
            cfgs.append((GroupLassoSolver, {'lam': float(lam), 'groups': make_groups(p, ng)}, f'GrpLasso λ={lam:.4g} G={ng}'))

    if kind == 'regression':
        lf1 = lam_grid(7); lf2 = lam_grid(8)
        for l1, l2 in itertools.product(lf1, lf2):
            cfgs.append((FusedLassoSolver, {'lam1': float(l1), 'lam2': float(l2)}, f'Fused λ1={l1:.4g} λ2={l2:.4g}'))

    for g in [0.001, 0.005, 0.01, 0.05, 0.1]:
        for T in [5, 10, 20, 30, 50, 80, 100, 150, 200, 300]:
            cfgs.append((SGDSolver, {'gamma0': g, 'n_passes': T}, f'SGD γ={g} T={T}'))
    for g in [0.001, 0.005, 0.01, 0.05, 0.1]:
        for a in [0.51, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
            cfgs.append((ImplicitSGDSolver, {'gamma0': g, 'alpha': a, 'n_passes': 10}, f'ISGD γ={g} α={a}'))
    for e in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for T in [5, 10, 20, 30, 50, 80, 100, 150, 200, 300]:
            cfgs.append((AdaGradSolver, {'eta': e, 'n_passes': T}, f'AdaGrad η={e} T={T}'))
    for lam in lam_grid(5):
        for g in [0.001, 0.01, 0.05, 0.1, 0.5]:
            for T in [10, 20]:
                cfgs.append((FOBOSSolver, {'lam': float(lam), 'gamma0': g, 'n_passes': T}, f'FOBOS λ={lam:.4g} γ={g} T={T}'))
    for lam in lam_grid(5):
        for g in [0.001, 0.01, 0.05, 0.1, 0.5]:
            for T in [5, 10]:
                cfgs.append((RDASolver, {'lam': float(lam), 'gamma0': g, 'n_passes': T}, f'RDA λ={lam:.4g} γ={g} T={T}'))
    for lam in lam_grid(5):
        for g in [0.001, 0.01, 0.05, 0.1, 0.5]:
            for K in [5, 10]:
                cfgs.append((TruncatedGradientSolver, {'lam': float(lam), 'gamma0': g, 'K': K}, f'TruncGrad λ={lam:.4g} γ={g} K={K}'))
    for nb_ in [3, 5, 10, 20, 50]:
        for ri in [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.3, 1.0, 3.0, 10.0]:
            cfgs.append((RenewableGLMSolver, {'n_batches': nb_, 'ridge_init': ri}, f'Renew B={nb_} r={ri}'))
    return cfgs

# ── Run arena ────────────────────────────────────────────────────────────────
def run_arena(datasets):
    records = []
    t0_total = time.time()
    for di, (ds_name, ds) in enumerate(datasets.items()):
        X, y, link, kind = ds['X'], ds['y'], ds['link'], ds['kind']
        n, p = X.shape
        lam_max = compute_lam_max(X, y)
        solver_cfgs = build_configs(lam_max, p, kind)
        t0 = time.time(); n_ok = 0
        for SolverCls, cfg, label in solver_cfgs:
            if kind == 'logistic' and SolverCls in SKIP_FOR_LOGISTIC:
                continue
            try:
                res = SolverCls(config=cfg).fit(X, y, link=link)
                beta = res.beta_hat
                if not np.all(np.isfinite(beta)):
                    raise ValueError('non-finite')
                records.append(dict(
                    dataset=ds_name, kind=kind, label=label,
                    cls=SolverCls.__name__, family=FAMILY[SolverCls],
                    beta=beta.copy(), bnorm=float(np.linalg.norm(beta)),
                    ok=True, p=p,
                ))
                n_ok += 1
            except Exception:
                pass
        print(f"  [{di+1:2d}/{len(datasets)}] {ds_name:35s}  n={n:4d} p={p:3d}  ok={n_ok:3d}  {time.time()-t0:.1f}s")
    print(f"\nArena done in {time.time()-t0_total:.1f}s — {len(records)} runs")
    return pd.DataFrame(records)

# ── Plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="'Alright Sans', 'Inter', 'Helvetica Neue', sans-serif",
              size=12, color='#1a1a1a'),
    margin=dict(l=0, r=0, t=40, b=0),
    hoverlabel=dict(
        bgcolor='#1a1a1a', font_color='#f5f5f5',
        font_family="'Alright Sans', 'Inter', sans-serif",
        font_size=12, bordercolor='#333',
    ),
)

_LEGEND_BASE = dict(
    bgcolor='rgba(255,255,255,0.9)',
    bordercolor='#e0e0e0', borderwidth=1,
    font=dict(size=11),
)

# ── t-SNE + UMAP figure ───────────────────────────────────────────────────────
def _sk_ver():
    return tuple(int(x) for x in sklearn.__version__.split('.')[:2])

def make_embedding_figure(df_ds, ds_name, ds_meta):
    """Interactive t-SNE + UMAP side-by-side for one dataset."""
    n, p = ds_meta['X'].shape
    kind = ds_meta['kind']

    betas = np.vstack(df_ds['beta'].values)
    # unit-normalise
    norms = np.linalg.norm(betas, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    B_norm = betas / norms

    # Outlier filter: remove runs with norm > 30× median
    raw_norms = np.linalg.norm(betas, axis=1)
    med = np.median(raw_norms)
    ok_mask = raw_norms <= 30 * med + 1e-9
    B_norm = B_norm[ok_mask]
    df_ok = df_ds[ok_mask].reset_index(drop=True)

    if len(df_ok) < 5:
        return None

    # t-SNE
    _iter_kw = 'max_iter' if _sk_ver() >= (1, 5) else 'n_iter'
    perp = min(30, len(df_ok) // 4)
    perp = max(5, perp)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, **{_iter_kw: 1000})
    Z_tsne = tsne.fit_transform(B_norm)

    # UMAP
    n_neighbors = min(15, len(df_ok) - 1)
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    Z_umap = reducer.fit_transform(B_norm)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('t-SNE', 'UMAP'),
        horizontal_spacing=0.08,
    )

    # Plot each solver as a separate trace (for legend toggle)
    solver_order = [s.__name__ for s in _SOLVER_ORDER]
    plotted = set()
    for solver_name in solver_order:
        mask = df_ok['cls'] == solver_name
        if not mask.any():
            continue
        sub = df_ok[mask]
        idxs = np.where(mask.values)[0]
        color = ALGO_COLORS.get(solver_name, '#888')
        label = ALGO_LABELS.get(solver_name, solver_name)
        family = sub['family'].iloc[0]

        # Hover text: "solver — config\n‖β‖ = X.XX"
        hover_texts = [
            f"<b>{label}</b><br>{row['label']}<br>‖β̂‖ = {row['bnorm']:.4f}"
            for _, row in sub.iterrows()
        ]

        show_legend = solver_name not in plotted
        plotted.add(solver_name)

        for col, Z in [(1, Z_tsne), (2, Z_umap)]:
            fig.add_trace(go.Scatter(
                x=Z[idxs, 0], y=Z[idxs, 1],
                mode='markers',
                marker=dict(
                    color=color, size=7, opacity=0.82,
                    line=dict(width=0.5, color='rgba(255,255,255,0.6)'),
                ),
                name=label,
                legendgroup=solver_name,
                showlegend=(show_legend and col == 1),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                customdata=sub['label'].values,
            ), row=1, col=col)
            show_legend = False

    # Style axes
    for col in [1, 2]:
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                         showline=False, row=1, col=col)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                         showline=False, row=1, col=col)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"<b>{ds_name.replace('_', ' ')}</b>  ·  n={n}, p={p}, {kind}  ·  {len(df_ok)} runs",
            font=dict(size=14, color='#1a1a1a'),
            x=0, xanchor='left',
        ),
        height=460,
        legend=dict(
            **_LEGEND_BASE,
            orientation='v',
            x=1.01, xanchor='left', y=0.5, yanchor='middle',
            itemsizing='constant',
        ),
        dragmode='pan',
    )

    return fig

# ── Heatmap figure ─────────────────────────────────────────────────────────────
def make_heatmap_figure(df_ds, ds_name, ds_meta):
    """Interactive coefficient heatmap for one dataset."""
    n, p = ds_meta['X'].shape

    _FAM_ORDER = {'Classical': 0, 'Penalized': 1, 'First-order': 2, 'Online': 3}
    df2 = df_ds.copy()
    df2['_fam_ord'] = df2['family'].map(_FAM_ORDER).fillna(9)
    df2 = df2.sort_values(['_fam_ord', 'cls', 'bnorm']).reset_index(drop=True)

    betas = np.vstack(df2['beta'].values)
    n_runs, _ = betas.shape

    # Widen single-config solvers: repeat their row 8× so it's visible
    single_config_solvers = {'OLSSolver', 'GLMIRLSSolver'}
    expanded_rows, expanded_labels, expanded_cls, expanded_families = [], [], [], []
    for i, row in df2.iterrows():
        repeat = 8 if row['cls'] in single_config_solvers else 1
        for _ in range(repeat):
            expanded_rows.append(betas[i])
            expanded_labels.append(row['label'])
            expanded_cls.append(row['cls'])
            expanded_families.append(row['family'])

    B = np.array(expanded_rows)

    # Subsample rows for high-dimensional data to keep HTML files loadable
    # Target: keep total cells (n_rows × p) below 80 000
    MAX_CELLS = 80_000
    if len(B) * p > MAX_CELLS:
        n_keep = max(120, MAX_CELLS // max(p, 1))
        rng = np.random.default_rng(42)
        families_arr = np.array(expanded_families)
        unique_fams = list(dict.fromkeys(expanded_families))
        keep_idx = []
        for fam in unique_fams:
            fam_idx = np.where(families_arr == fam)[0]
            n_fam = max(1, round(len(fam_idx) * n_keep / len(B)))
            chosen = sorted(rng.choice(fam_idx, min(n_fam, len(fam_idx)), replace=False).tolist())
            keep_idx.extend(chosen)
        keep_idx = sorted(keep_idx)
        expanded_rows     = [expanded_rows[i]     for i in keep_idx]
        expanded_labels   = [expanded_labels[i]   for i in keep_idx]
        expanded_cls      = [expanded_cls[i]      for i in keep_idx]
        expanded_families = [expanded_families[i] for i in keep_idx]
        B = np.array(expanded_rows)

    # Compact per-row hover text (same label for every column in a row → much smaller JSON)
    hover_cls = [ALGO_LABELS.get(c, c) for c in expanded_cls]
    row_labels = [f"<b>{hover_cls[r]}</b>  {expanded_labels[r]}" for r in range(len(B))]
    hover_text = [[f"{row_labels[r]}<br>j={j}  β={B[r,j]:.4f}" for j in range(p)] for r in range(len(B))]

    vmax = float(np.percentile(np.abs(B), 99))
    vmax = max(vmax, 1e-6)

    # y-tick labels at midpoints of each solver block
    y_tickvals, y_ticktext = [], []
    pos = 0
    prev_cls = None
    block_start = 0
    for i, cls_name in enumerate(expanded_cls + [None]):
        if cls_name != prev_cls and prev_cls is not None:
            mid = (block_start + i - 1) / 2.0
            y_tickvals.append(mid)
            y_ticktext.append(ALGO_LABELS.get(prev_cls, prev_cls))
            block_start = i
        prev_cls = cls_name
        pos = i

    # Colour strip on left (as annotation band)
    strip_colors = [ALGO_COLORS.get(c, '#888') for c in expanded_cls]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=B,
        colorscale=[
            [0.0, '#053061'], [0.1, '#2166ac'], [0.2, '#4393c3'],
            [0.3, '#92c5de'], [0.4, '#d1e5f0'], [0.5, '#f7f7f7'],
            [0.6, '#fddbc7'], [0.7, '#f4a582'], [0.8, '#d6604d'],
            [0.9, '#b2182b'], [1.0, '#67001f'],
        ],
        zmin=-vmax, zmax=vmax,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title=dict(text='β̂', side='right', font=dict(size=12)),
            thickness=12, len=0.8,
            tickfont=dict(size=10),
            bgcolor='rgba(0,0,0,0)',
        ),
        xgap=0, ygap=0,
    ))

    # Horizontal divider lines between solver families
    shape_list = []
    pos = 0
    prev_fam = None
    for i, fam in enumerate(expanded_families + [None]):
        if fam != prev_fam and prev_fam is not None:
            shape_list.append(dict(
                type='line', xref='paper', yref='y',
                x0=0, x1=1, y0=pos - 0.5, y1=pos - 0.5,
                line=dict(color='rgba(255,255,255,0.7)', width=1.5),
            ))
        prev_fam = fam
        pos = i

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"<b>{ds_name.replace('_', ' ')}</b>  ·  n={n}, p={p}  ·  coefficient heatmap",
            font=dict(size=14, color='#1a1a1a'), x=0, xanchor='left',
        ),
        height=max(350, min(len(B) * 4 + 80, 900)),
        xaxis=dict(
            title='Feature index j', tickfont=dict(size=10),
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            tickmode='array', tickvals=y_tickvals, ticktext=y_ticktext,
            tickfont=dict(size=10), showgrid=False, zeroline=False,
            autorange='reversed',
        ),
        shapes=shape_list,
        dragmode='zoom',
    )

    return fig

# ── Write single HTML figure ──────────────────────────────────────────────────
PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

def write_figure_html(fig, path):
    """Write a self-contained figure HTML with transparent background."""
    html = fig.to_html(
        full_html=True,
        include_plotlyjs='cdn',
        config=dict(
            displayModeBar=True,
            modeBarButtonsToRemove=['select2d', 'lasso2d', 'autoScale2d'],
            displaylogo=False,
            scrollZoom=True,
            toImageButtonOptions=dict(format='svg', filename=os.path.splitext(os.path.basename(path))[0]),
        ),
    )
    # Make page background transparent so iframe blends with site
    html = html.replace(
        '<body>',
        '<body style="margin:0;padding:0;background:transparent;overflow:hidden;">'
    ).replace(
        '<head>',
        '<head><style>body,html{background:transparent!important;} .plotly-graph-div{background:transparent!important;}</style>'
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)

# ── Dataset metadata JSON ─────────────────────────────────────────────────────
DS_DESCRIPTIONS = {
    'Diabetes':             ('UCI / sklearn', 'Predict disease progression from 10 physiological features. Classic regression benchmark.'),
    'Breast_Cancer':        ('UCI / sklearn', 'Binary classification: malignant vs benign tumour from 30 cell nucleus features.'),
    'Digits_hi5':           ('NIST / sklearn', 'Handwritten digit images (8×8 pixels); binary task: digit ≥ 5.'),
    'Fair_Affairs':         ('statsmodels', 'Predict number of extra-marital affairs from socio-demographic variables.'),
    'RAND_HIE':             ('statsmodels', 'RAND Health Insurance Experiment: predict log medical expenditure.'),
    'STAR98_pass':          ('statsmodels', 'Texas school performance data — predict pass rate above median.'),
    'ANES96_vote':          ('statsmodels', 'American National Election Study 1996 — predict vote choice.'),
    'Mode_Choice':          ('statsmodels', 'Travel mode choice (car vs public transit) from trip and person attributes.'),
    'Synth_Reg_Sparse200':  ('synthetic', 'n=2000, p=200, 20 informative features — sparse high-dimensional regression.'),
    'Synth_Reg_Dense100':   ('synthetic', 'n=2000, p=100, all features informative — dense regression.'),
    'Synth_Reg_Corr150':    ('synthetic', 'n=2000, p=150, effective rank 10 — strongly multicollinear regression.'),
    'Synth_Reg_HighDim300': ('synthetic', 'n=2000, p=300, 30 informative — very high-dimensional sparse regression.'),
    'Synth_Friedman1':      ('synthetic', 'Friedman #1 nonlinear ground truth, p=10 — baseline with nonlinear structure.'),
    'Synth_Logit_Dense100': ('synthetic', 'n=2000, p=100, 80 informative — dense logistic classification.'),
    'Synth_Logit_Sparse200':('synthetic', 'n=2000, p=200, 25 informative — sparse logistic classification.'),
    'Synth_Logit_Noisy150': ('synthetic', 'n=2000, p=150, 12% label noise — noisy logistic classification.'),
}

# ── Main ──────────────────────────────────────────────────────────────────────
def main(no_cache=False):
    # 1. Load or compute arena results
    if not no_cache and os.path.exists(CACHE_PATH):
        print("Loading cached arena results …")
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        df_all = cache['df_all']
        datasets = cache['datasets']
        print(f"  Loaded {len(df_all)} records, {len(datasets)} datasets.")
    else:
        print("Loading datasets …")
        datasets = load_datasets()
        print(f"  {len(datasets)} datasets loaded.")
        print("\nRunning arena (this may take 10-15 min) …")
        df_all = run_arena(datasets)
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump({'df_all': df_all, 'datasets': datasets}, f)
        print(f"  Cached to {CACHE_PATH}")

    # Filter outliers (30× median norm) — avoid groupby().apply() pandas-2 breakage
    df_ok = df_all[df_all['ok']].copy()
    med_bnorm = df_ok.groupby('dataset')['bnorm'].median().rename('_med')
    df_ok = df_ok.join(med_bnorm, on='dataset')
    df_clean = df_ok[df_ok['bnorm'] <= 30 * df_ok['_med'] + 1e-9].drop(columns='_med').reset_index(drop=True)

    # 2. Generate figures
    ds_list = []
    for ds_name, ds_meta in datasets.items():
        df_ds = df_clean[df_clean['dataset'] == ds_name].copy()
        if len(df_ds) < 5:
            continue

        kind = ds_meta['kind']
        n, p = ds_meta['X'].shape
        source, desc = DS_DESCRIPTIONS.get(ds_name, ('', ''))
        ds_list.append({
            'name': ds_name, 'kind': kind, 'n': n, 'p': p,
            'source': source, 'description': desc,
            'n_runs': int(len(df_ds)),
            'embed_file': f'figures/tsne_umap_{ds_name}.html',
            'heatmap_file': f'figures/heatmap_{ds_name}.html',
        })

        # Embedding figure
        embed_path = os.path.join(OUT_DIR, f'tsne_umap_{ds_name}.html')
        print(f"  Embedding: {ds_name} …", end=' ', flush=True)
        fig_e = make_embedding_figure(df_ds, ds_name, ds_meta)
        if fig_e:
            write_figure_html(fig_e, embed_path)
            print("OK")
        else:
            print("SKIP (too few points)")

        # Heatmap figure
        heatmap_path = os.path.join(OUT_DIR, f'heatmap_{ds_name}.html')
        print(f"  Heatmap:   {ds_name} …", end=' ', flush=True)
        fig_h = make_heatmap_figure(df_ds, ds_name, ds_meta)
        if fig_h:
            write_figure_html(fig_h, heatmap_path)
            print("OK")
        else:
            print("SKIP")

    # 3. Write datasets metadata JSON
    meta_path = os.path.join(OUT_DIR, 'datasets.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(ds_list, f, indent=2)
    print(f"\n  Wrote metadata: {meta_path}")
    print(f"  Wrote {len(ds_list)*2} HTML figures to {OUT_DIR}")
    print("\nDone. Run `mkdocs build` or `mkdocs serve` to see the Arena pages.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cache', action='store_true', help='Force re-run even if cache exists')
    args = parser.parse_args()
    main(no_cache=args.no_cache)
