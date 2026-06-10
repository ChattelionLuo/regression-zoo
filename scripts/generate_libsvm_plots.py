"""
generate_libsvm_plots.py
========================
Generates interactive Plotly HTML figures for the LIBSVM datasets in the Arena.

Reads:  notebooks/libsvm_arena_results.pkl
Writes: docs/arena/figures/tsne_umap_libsvm_<dataset>.html
        docs/arena/figures/heatmap_libsvm_<dataset>.html

Usage:
    python scripts/generate_libsvm_plots.py
"""
from __future__ import annotations
import sys, os, warnings, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
import sklearn

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed — UMAP subplot will be skipped.")

# ── Paths ─────────────────────────────────────────────────────────────────────
PKL_PATH = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'libsvm_arena_results.pkl')
OUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'docs', 'arena', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Solver display metadata ───────────────────────────────────────────────────
ALGO_LABELS = {
    "OLSSolver": "OLS", "GLMIRLSSolver": "GLM-IRLS", "LARSSolver": "LARS",
    "RidgeSolver": "Ridge", "LassoCDSolver": "Lasso-CD",
    "ElasticNetSolver": "Elastic Net", "AdaptiveLassoSolver": "Adaptive Lasso",
    "SCADLLASolver": "SCAD", "MCPCDSolver": "MCP-CD",
    "GroupLassoSolver": "Group Lasso",
    "ISTASolver": "ISTA", "FISTASolver": "FISTA",
    "SGDSolver": "SGD", "ImplicitSGDSolver": "Implicit SGD",
    "AdaGradSolver": "AdaGrad", "FOBOSSolver": "FOBOS",
    "RDASolver": "RDA", "RenewableGLMSolver": "Renewable GLM",
}

_SOLVER_ORDER = [
    "OLSSolver", "GLMIRLSSolver", "LARSSolver",
    "RidgeSolver", "LassoCDSolver", "ElasticNetSolver",
    "AdaptiveLassoSolver", "SCADLLASolver", "MCPCDSolver", "GroupLassoSolver",
    "ISTASolver", "FISTASolver",
    "SGDSolver", "ImplicitSGDSolver", "AdaGradSolver",
    "FOBOSSolver", "RDASolver", "RenewableGLMSolver",
]

_PALETTE = [
    "#1a1a2e", "#16213e", "#0f3460", "#533483",
    "#2d6a4f", "#40916c", "#74c69d", "#b7e4c7",
    "#7b2d8b", "#b5179e", "#f72585", "#ff6b6b",
    "#ee9b00", "#ca6702", "#bb3e03", "#ae2012",
    "#005f73", "#0a9396", "#94d2bd", "#e9d8a6",
]
ALGO_COLORS = {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(_SOLVER_ORDER)}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, system-ui, sans-serif', size=11, color='#1a1a2e'),
)
_LEGEND_BASE = dict(
    bgcolor='rgba(255,255,255,0.85)',
    bordercolor='rgba(0,0,0,0.08)',
    borderwidth=1,
    font=dict(size=10),
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _sk_ver():
    return tuple(int(x) for x in sklearn.__version__.split('.')[:2])


def make_embedding_figure(df_ds: pd.DataFrame, ds_name: str, p: int, link: str) -> go.Figure | None:
    """t-SNE + UMAP scatter of beta vectors coloured by solver."""
    betas = np.vstack(df_ds['beta'].values)

    # Unit-normalise
    norms = np.linalg.norm(betas, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    B_norm = betas / norms

    # Outlier filter
    raw_norms = np.linalg.norm(betas, axis=1)
    med = np.median(raw_norms)
    ok_mask = raw_norms <= 30 * med + 1e-9
    B_norm = B_norm[ok_mask]
    df_ok = df_ds[ok_mask].reset_index(drop=True)

    if len(df_ok) < 5:
        return None

    # t-SNE
    _iter_kw = 'max_iter' if _sk_ver() >= (1, 5) else 'n_iter'
    perp = max(5, min(30, len(df_ok) // 4))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, **{_iter_kw: 1000})
    Z_tsne = tsne.fit_transform(B_norm)

    if HAS_UMAP:
        n_neighbors = min(15, len(df_ok) - 1)
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors,
                                 min_dist=0.1, random_state=42)
        Z_umap = reducer.fit_transform(B_norm)
        fig = make_subplots(rows=1, cols=2, subplot_titles=('t-SNE', 'UMAP'),
                            horizontal_spacing=0.03)
        all_Z = [(1, Z_tsne), (2, Z_umap)]
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=('t-SNE',))
        all_Z = [(1, Z_tsne)]

    plotted = set()
    for solver_name in _SOLVER_ORDER:
        mask = df_ok['cls'] == solver_name
        if not mask.any():
            continue
        sub = df_ok[mask]
        idxs = np.where(mask.values)[0]
        color = ALGO_COLORS.get(solver_name, '#888')
        label = ALGO_LABELS.get(solver_name, solver_name)
        hover_texts = [
            f"<b>{label}</b><br>{row['label']}<br>‖β̂‖ = {row['bnorm']:.4f}"
            for _, row in sub.iterrows()
        ]
        show_legend = solver_name not in plotted
        plotted.add(solver_name)

        for col, Z in all_Z:
            fig.add_trace(go.Scatter(
                x=Z[idxs, 0], y=Z[idxs, 1],
                mode='markers',
                marker=dict(color=color, size=7, opacity=0.82,
                            line=dict(width=0.5, color='rgba(255,255,255,0.6)')),
                name=label,
                legendgroup=solver_name,
                showlegend=(show_legend and col == 1),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
            ), row=1, col=col)
            show_legend = False

    for col in range(1, len(all_Z) + 1):
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                         showline=False, row=1, col=col)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                         showline=False, row=1, col=col)

    fig.update_layout(
        **{**PLOTLY_LAYOUT, 'margin': dict(l=0, r=0, t=26, b=0)},
        title=None, height=460,
        legend=dict(**_LEGEND_BASE, orientation='v',
                    x=1.01, xanchor='left', y=0.5, yanchor='middle',
                    itemsizing='constant', tracegroupgap=2),
        dragmode='pan',
    )
    return fig


def make_heatmap_figure(df_ds: pd.DataFrame, ds_name: str, p: int) -> go.Figure | None:
    """Coefficient heatmap with mean-|β̂| bar, for one dataset."""
    _FAM_ORDER = {'Classical': 0, 'Penalized': 1, 'First-order': 2, 'Online': 3}
    df2 = df_ds.copy()
    df2['_fam_ord'] = df2['family'].map(_FAM_ORDER).fillna(9)
    df2 = df2.sort_values(['_fam_ord', 'cls', 'bnorm']).reset_index(drop=True)

    betas = np.vstack(df2['beta'].values)

    # Widen single-config solvers
    single_config = {'OLSSolver', 'GLMIRLSSolver'}
    exp_rows, exp_labels, exp_cls, exp_families = [], [], [], []
    for i, row in df2.iterrows():
        repeat = 8 if row['cls'] in single_config else 1
        for _ in range(repeat):
            exp_rows.append(betas[i])
            exp_labels.append(row['label'])
            exp_cls.append(row['cls'])
            exp_families.append(row['family'])

    B = np.array(exp_rows)

    # Subsample if too large
    MAX_CELLS = 60_000
    if len(B) * p > MAX_CELLS:
        n_keep = max(100, MAX_CELLS // max(p, 1))
        rng = np.random.default_rng(42)
        fam_arr = np.array(exp_families)
        unique_fams = list(dict.fromkeys(exp_families))
        keep_idx = []
        for fam in unique_fams:
            fam_idx = np.where(fam_arr == fam)[0]
            n_fam = max(1, round(len(fam_idx) * n_keep / len(B)))
            chosen = sorted(rng.choice(fam_idx, min(n_fam, len(fam_idx)), replace=False).tolist())
            keep_idx.extend(chosen)
        keep_idx = sorted(keep_idx)
        exp_rows = [exp_rows[i] for i in keep_idx]
        exp_labels = [exp_labels[i] for i in keep_idx]
        exp_cls = [exp_cls[i] for i in keep_idx]
        exp_families = [exp_families[i] for i in keep_idx]
        B = np.array(exp_rows)

    if len(B) == 0:
        return None

    n_rows = len(B)
    hover_cls = [ALGO_LABELS.get(c, c) for c in exp_cls]
    row_y_labels = [f"{hover_cls[r]}  ·  {exp_labels[r]}" for r in range(n_rows)]

    vmax = float(np.percentile(np.abs(B), 99))
    vmax = max(vmax, 1e-6)
    mean_abs_beta = np.mean(np.abs(B), axis=0)

    # y-tick labels at block midpoints
    y_tickvals, y_ticktext = [], []
    block_start, prev_cls = 0, None
    for i, cls_name in enumerate(exp_cls + [None]):
        if cls_name != prev_cls and prev_cls is not None:
            mid_idx = (block_start + i - 1) // 2
            y_tickvals.append(row_y_labels[mid_idx])
            y_ticktext.append(ALGO_LABELS.get(prev_cls, prev_cls))
            block_start = i
        prev_cls = cls_name

    # Family divider shapes
    divider_shapes = []
    prev_fam = None
    for i, fam in enumerate(exp_families + [None]):
        if fam != prev_fam and prev_fam is not None:
            y_ref = row_y_labels[i - 1] if i < n_rows else row_y_labels[-1]
            divider_shapes.append(dict(
                type='line', xref='x', yref='y2',
                x0=-0.5, x1=p - 0.5, y0=y_ref, y1=y_ref,
                line=dict(color='rgba(200,200,200,0.5)', width=1),
            ))
        prev_fam = fam

    # x-tick step
    if p <= 20:      xtick_step = 1
    elif p <= 64:    xtick_step = 5
    elif p <= 150:   xtick_step = 10
    elif p <= 500:   xtick_step = 25
    elif p <= 2000:  xtick_step = 100
    else:            xtick_step = 500
    xtick_vals = list(range(0, p, xtick_step))

    fig = make_subplots(rows=2, cols=1, row_heights=[0.12, 0.88],
                        shared_xaxes=True, vertical_spacing=0.02)

    fig.add_trace(go.Bar(
        x=list(range(p)), y=mean_abs_beta.tolist(),
        marker_color='#4a6fa5', marker_line_width=0, opacity=0.85,
        hovertemplate='j=%{x}<br>mean|β̂|=%{y:.4f}<extra></extra>',
        showlegend=False, name='',
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=B, y=row_y_labels,
        colorscale=[
            [0.0, '#053061'], [0.1, '#2166ac'], [0.2, '#4393c3'],
            [0.3, '#92c5de'], [0.4, '#d1e5f0'], [0.5, '#f7f7f7'],
            [0.6, '#fddbc7'], [0.7, '#f4a582'], [0.8, '#d6604d'],
            [0.9, '#b2182b'], [1.0, '#67001f'],
        ],
        zmin=-vmax, zmax=vmax,
        hovertemplate='<b>%{y}</b><br>j=%{x}  β̂=%{z:.4f}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title=dict(text='β̂', side='right', font=dict(size=11)),
            thickness=10, len=0.75,
            tickfont=dict(size=9),
            bgcolor='rgba(0,0,0,0)',
            y=0.44, yanchor='middle',
        ),
        xgap=0, ygap=0,
    ), row=2, col=1)

    heatmap_height = max(300, min(n_rows * 3 + 80, 820))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=None,
        height=heatmap_height + 80,
        shapes=divider_shapes,
        dragmode='zoom',
        bargap=0,
    )
    fig.update_yaxes(title=dict(text='mean |β̂|', font=dict(size=9)),
                     tickfont=dict(size=9), showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(tickmode='array', tickvals=y_tickvals, ticktext=y_ticktext,
                     tickfont=dict(size=10), showgrid=False, zeroline=False,
                     autorange='reversed', row=2, col=1)
    fig.update_xaxes(
        title=dict(text='feature index j', font=dict(size=10)),
        tickmode='array', tickvals=xtick_vals,
        ticktext=[str(v) for v in xtick_vals],
        tickfont=dict(size=9), showgrid=False, zeroline=False,
        row=2, col=1,
    )
    return fig


def write_figure_html(fig: go.Figure, path: str) -> None:
    html = fig.to_html(
        full_html=True, include_plotlyjs='cdn',
        config=dict(
            displayModeBar=True,
            modeBarButtonsToRemove=['select2d', 'lasso2d', 'autoScale2d'],
            displaylogo=False,
            scrollZoom=True,
            toImageButtonOptions=dict(format='svg',
                                      filename=os.path.splitext(os.path.basename(path))[0]),
        ),
    )
    html = html.replace(
        '<body>', '<body style="margin:0;padding:0;background:transparent;overflow:hidden;">'
    ).replace(
        '<head>',
        '<head><style>body,html{background:transparent!important;} '
        '.plotly-graph-div{background:transparent!important;}</style>'
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {PKL_PATH} …")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
    df_all = data['df_all']
    ds_meta = data.get('datasets', {})

    print(f"  {len(df_all)} rows, {df_all['dataset'].nunique()} datasets\n")

    datasets = sorted(df_all['dataset'].unique())
    for ds_name in datasets:
        df_ds = df_all[df_all['dataset'] == ds_name].copy()
        n_runs = len(df_ds)
        p = int(df_ds['p'].iloc[0])
        n = int(df_ds['n'].iloc[0])
        link = df_ds['link'].iloc[0]
        print(f"  {ds_name:25s}  n={n:6d}  p={p:6d}  runs={n_runs:4d}", end='  ')

        # Embedding map
        emb_path = os.path.join(OUT_DIR, f'tsne_umap_libsvm_{ds_name}.html')
        if n_runs >= 5:
            fig = make_embedding_figure(df_ds, ds_name, p, link)
            if fig:
                write_figure_html(fig, emb_path)
                print('emb OK', end='  ')
            else:
                print('emb --', end='  ')
        else:
            print('emb --', end='  ')

        # Heatmap
        hm_path = os.path.join(OUT_DIR, f'heatmap_libsvm_{ds_name}.html')
        if n_runs >= 1:
            fig = make_heatmap_figure(df_ds, ds_name, p)
            if fig:
                write_figure_html(fig, hm_path)
                print('hm OK')
            else:
                print('hm --')
        else:
            print('hm --')

    print(f"\nDone — figures written to {OUT_DIR}")


if __name__ == '__main__':
    main()
