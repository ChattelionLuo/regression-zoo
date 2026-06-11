"""
libsvm_arena_v2.py  —  Unified LIBSVM arena runner (v2)
=========================================================
Matches the original arena protocol exactly:
  • StandardScaler(X) for dense; MaxAbsScaler(X) for sparse
  • y standardised for regression (zero-mean, unit-variance)
  • y ∈ {0,1} for binary classification
  • lam_max computed from the preprocessed data
  • 50 data-driven, log-spaced hyperparameter configs per solver
  • Same config keys as the original (gamma0, n_passes, eta, …)
  • Same feasibility filters (n×p, p, sparse, link)
  • Supports incremental resume: skips datasets already saved

Usage (single dataset):
    python scripts/libsvm_arena_v2.py --dataset abalone

Usage (all datasets, useful for testing locally):
    python scripts/libsvm_arena_v2.py --all

Results saved per-dataset to:
    notebooks/libsvm_v2/libsvm_v2_{dataset}.pkl

Merge with:
    python scripts/libsvm_arena_v2_merge.py
"""
from __future__ import annotations
import sys, os, argparse, bz2, gzip, time, pickle, warnings
warnings.filterwarnings('ignore')  # suppress sklearn ConvergenceWarning etc.
import numpy as np
import scipy.sparse as sp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.datasets import load_svmlight_file

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR  = Path('/home/23038969r/libsvm_data')
OUT_DIR   = BASE_DIR / 'notebooks' / 'libsvm_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# On local machine fall back to a sibling data dir
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR / 'notebooks' / 'libsvm_data'
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset list ──────────────────────────────────────────────────────────────
BASE_R = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression'
BASE_B = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary'

DATASETS = [
    # ── Regression ──────────────────────────────────────────────────────────
    ("abalone",      f"{BASE_R}/abalone",                               "regression"),
    ("bodyfat",      f"{BASE_R}/bodyfat",                               "regression"),
    ("cadata",       f"{BASE_R}/cadata",                                "regression"),
    ("cpusmall",     f"{BASE_R}/cpusmall_scale",                        "regression"),
    ("eunite2001",   f"{BASE_R}/eunite2001",                            "regression"),
    ("housing",      f"{BASE_R}/housing",                               "regression"),
    ("mg",           f"{BASE_R}/mg",                                    "regression"),
    ("mpg",          f"{BASE_R}/mpg",                                   "regression"),
    ("pyrim",        f"{BASE_R}/pyrim",                                 "regression"),
    ("space_ga",     f"{BASE_R}/space_ga",                              "regression"),
    ("triazines",    f"{BASE_R}/triazines",                             "regression"),
    ("YearPredMSD",  f"{BASE_R}/YearPredictionMSD.bz2",                 "regression"),
    # ── Binary classification ────────────────────────────────────────────────
    ("australian",   f"{BASE_B}/australian",                            "binary"),
    ("breast-cancer",f"{BASE_B}/breast-cancer",                         "binary"),
    ("cod-rna",      f"{BASE_B}/cod-rna",                               "binary"),
    ("colon-cancer", f"{BASE_B}/colon-cancer.bz2",                      "binary"),
    ("diabetes",     f"{BASE_B}/diabetes",                              "binary"),
    ("duke",         f"{BASE_B}/duke.bz2",                              "binary"),
    ("fourclass",    f"{BASE_B}/fourclass",                             "binary"),
    ("german-numer", f"{BASE_B}/german.numer",                          "binary"),
    ("gisette",      f"{BASE_B}/gisette_scale.bz2",                     "binary"),
    ("heart",        f"{BASE_B}/heart",                                 "binary"),
    ("ijcnn1",       f"{BASE_B}/ijcnn1.bz2",                            "binary"),
    ("ionosphere",   f"{BASE_B}/ionosphere_scale",                      "binary"),
    ("leukemia",     f"{BASE_B}/leu.bz2",                               "binary"),
    ("liver-disorders", f"{BASE_B}/liver-disorders",                    "binary"),
    ("madelon",      f"{BASE_B}/madelon",                               "binary"),
    ("mushrooms",    f"{BASE_B}/mushrooms",                             "binary"),
    ("phishing",     f"{BASE_B}/phishing",                              "binary"),
    ("skin-nonskin", f"{BASE_B}/skin_nonskin",                          "binary"),
    ("sonar",        f"{BASE_B}/sonar_scale",                           "binary"),
    ("splice",       f"{BASE_B}/splice",                                "binary"),
    ("svmguide1",    f"{BASE_B}/svmguide1",                             "binary"),
    ("svmguide3",    f"{BASE_B}/svmguide3",                             "binary"),
    ("w8a",          f"{BASE_B}/w8a",                                   "binary"),
    ("a9a",          f"{BASE_B}/a9a",                                   "binary"),
    ("covtype-binary", f"{BASE_B}/covtype.libsvm.binary.scale.bz2",     "binary"),
    ("epsilon",      f"{BASE_B}/epsilon_normalized.bz2",                "binary"),
]

# ── Preprocessing limits ──────────────────────────────────────────────────────
MAX_N        = 80_000    # subsample rows
MAX_P_DENSE  = 10_000   # convert sparse→dense if p ≤ this
SEED         = 42
rng_global   = np.random.default_rng(SEED)

# ── Feasibility thresholds ────────────────────────────────────────────────────
# (same logic as before, but now applied against unified config list)
MAX_NP_EXPENSIVE  = 5_000_000   # LARS, GroupLasso, SCAD, MCP: skip dense when n*p > this
MAX_P_CUBIC       = 2_000       # ISTA, FISTA, GLMIRLS, RenewableGLM: skip dense p > this
MAX_NP_SAGA_LOGIT = 5_000_000   # LassoCD, ElasticNet SAGA logit: skip dense logit n*p > this
MAX_P_SPARSE_OPS  = 10_000      # Ridge, ISTA, FISTA, GLMIRLS: skip sparse p > this

DENSE_ONLY    = {'LARSSolver', 'GroupLassoSolver', 'SCADLLASolver',
                 'MCPCDSolver', 'AdaptiveLassoSolver', 'FusedLassoSolver'}
CUBIC_SOLVERS = {'ISTASolver', 'FISTASolver', 'GLMIRLSSolver', 'RenewableGLMSolver'}
SPARSE_OPS_SKIP = {'ISTASolver', 'FISTASolver', 'RidgeSolver',
                   'GLMIRLSSolver', 'RenewableGLMSolver'}
EXPENSIVE_DENSE = {'LARSSolver', 'GroupLassoSolver', 'SCADLLASolver', 'MCPCDSolver'}
SAGA_LOGIT    = {'LassoCDSolver', 'ElasticNetSolver'}
REGRESSION_ONLY = {'OLSSolver', 'LARSSolver', 'FusedLassoSolver'}  # skip for logit

# ── Download helper ───────────────────────────────────────────────────────────
def _download(name: str, url: str) -> Path | None:
    final = DATA_DIR / name
    if final.exists() and final.stat().st_size > 200:
        return final
    is_bz = url.endswith('.bz2')
    is_gz  = url.endswith('.gz')
    raw = DATA_DIR / (name + ('.bz2' if is_bz else '.gz' if is_gz else ''))
    if not raw.exists() or raw.stat().st_size < 200:
        print(f"  wget {name}…", flush=True)
        ret = os.system(f'wget -q -t 5 --timeout=120 -O "{raw}" "{url}"')
        if ret != 0 or not raw.exists() or raw.stat().st_size < 100:
            print(f"  WARN: wget failed for {name}", flush=True)
            return None
    if is_bz:
        print(f"  decompress {name}.bz2…", flush=True)
        with bz2.open(raw, 'rb') as fin, open(final, 'wb') as fout:
            fout.write(fin.read())
    elif is_gz:
        with gzip.open(raw, 'rb') as fin, open(final, 'wb') as fout:
            fout.write(fin.read())
    return final

# ── Load and preprocess one dataset ──────────────────────────────────────────
def load_dataset(name: str, url: str, ds_type: str):
    path = _download(name, url)
    if path is None:
        return None
    try:
        X, y = load_svmlight_file(str(path))
    except Exception as e:
        print(f"  SKIP {name}: {e}", flush=True)
        return None

    n_orig, p = X.shape

    # Binary: remap to {0, 1}
    if ds_type == 'binary':
        labels = np.unique(y)
        if len(labels) != 2:
            print(f"  SKIP {name}: {len(labels)} classes", flush=True)
            return None
        y = (y == labels[1]).astype(np.float64)
    else:
        y = y.astype(np.float64)

    # Subsample large datasets
    if n_orig > MAX_N:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(n_orig, MAX_N, replace=False)
        X = X[idx]; y = y[idx]
    n = len(y)

    # Decide dense vs sparse
    is_sparse = sp.issparse(X) and p > MAX_P_DENSE
    if sp.issparse(X) and not is_sparse:
        X = X.toarray()

    # Scale X
    if is_sparse:
        scaler_X = MaxAbsScaler()
    else:
        scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    if not is_sparse:
        X = np.asarray(X, dtype=np.float64)

    # ── KEY FIX: standardise y for regression ─────────────────────────────
    y_mean, y_std = 0.0, 1.0
    if ds_type == 'regression':
        y_mean = float(np.mean(y))
        y_std  = float(np.std(y))
        if y_std < 1e-12:
            y_std = 1.0
        y = (y - y_mean) / y_std

    link = 'logit' if ds_type == 'binary' else 'identity'
    return dict(
        name=name, X=X, y=y, link=link,
        n=int(n), p=int(p), is_sparse=is_sparse,
        y_mean=y_mean, y_std=y_std, source='LIBSVM', type=ds_type,
    )

# ── Lambda max (from preprocessed data) ──────────────────────────────────────
def compute_lam_max(X, y) -> float:
    y_c = y - float(np.mean(y))
    if sp.issparse(X):
        grad = np.abs(np.asarray(X.T.dot(y_c)).ravel()) / len(y)
    else:
        grad = np.abs(X.T @ y_c) / len(y)
    return float(grad.max())

# ── Solver imports ────────────────────────────────────────────────────────────
def load_solvers():
    from src.glmzoo.solvers.classical.ols          import OLSSolver
    from src.glmzoo.solvers.classical.glm_irls     import GLMIRLSSolver
    from src.glmzoo.solvers.classical.ridge        import RidgeSolver
    from src.glmzoo.solvers.penalized.lasso_cd     import LassoCDSolver
    from src.glmzoo.solvers.penalized.elastic_net  import ElasticNetSolver
    from src.glmzoo.solvers.penalized.adaptive_lasso import AdaptiveLassoSolver
    from src.glmzoo.solvers.penalized.scad_lla     import SCADLLASolver
    from src.glmzoo.solvers.penalized.mcp_cd       import MCPCDSolver
    from src.glmzoo.solvers.penalized.group_lasso  import GroupLassoSolver
    from src.glmzoo.solvers.path.lars              import LARSSolver
    from src.glmzoo.solvers.first_order.ista       import ISTASolver
    from src.glmzoo.solvers.first_order.fista      import FISTASolver
    from src.glmzoo.solvers.online.sgd             import SGDSolver
    from src.glmzoo.solvers.online.implicit_sgd    import ImplicitSGDSolver
    from src.glmzoo.solvers.online.adagrad         import AdaGradSolver
    from src.glmzoo.solvers.online.rda             import RDASolver
    from src.glmzoo.solvers.online.fobos           import FOBOSSolver
    from src.glmzoo.solvers.online.truncated_gradient import TruncatedGradientSolver
    from src.glmzoo.solvers.online.renewable_glm   import RenewableGLMSolver
    # FusedLasso only for regression
    try:
        from src.glmzoo.solvers.penalized.fused_lasso import FusedLassoSolver
    except ImportError:
        FusedLassoSolver = None
    solvers = [
        OLSSolver, GLMIRLSSolver, LARSSolver,
        RidgeSolver, LassoCDSolver, ElasticNetSolver,
        AdaptiveLassoSolver, SCADLLASolver, MCPCDSolver, GroupLassoSolver,
        ISTASolver, FISTASolver,
        SGDSolver, ImplicitSGDSolver, AdaGradSolver,
        FOBOSSolver, RDASolver, TruncatedGradientSolver, RenewableGLMSolver,
    ]
    if FusedLassoSolver is not None:
        solvers.insert(solvers.index(LARSSolver) + 1, FusedLassoSolver)
    return solvers

FAMILY = {
    'OLSSolver': 'Classical', 'GLMIRLSSolver': 'Classical', 'LARSSolver': 'Classical',
    'RidgeSolver': 'Penalized', 'LassoCDSolver': 'Penalized',
    'ElasticNetSolver': 'Penalized', 'AdaptiveLassoSolver': 'Penalized',
    'SCADLLASolver': 'Penalized', 'MCPCDSolver': 'Penalized',
    'GroupLassoSolver': 'Penalized', 'FusedLassoSolver': 'Penalized',
    'ISTASolver': 'First-order', 'FISTASolver': 'First-order',
    'SGDSolver': 'Online', 'ImplicitSGDSolver': 'Online', 'AdaGradSolver': 'Online',
    'FOBOSSolver': 'Online', 'RDASolver': 'Online',
    'TruncatedGradientSolver': 'Online', 'RenewableGLMSolver': 'Online',
}

# ── Config builder — matching original protocol exactly ───────────────────────
def build_configs(lam_max: float, p: int, link: str) -> list[tuple]:
    """Return list of (SolverCls, cfg_dict, label) matching original protocol."""
    # All solver classes (loaded once globally)
    global _SOLVERS
    Solvers = {cls.__name__: cls for cls in _SOLVERS}

    def lam_grid(n=50, frac_min=0.001):
        lo = max(lam_max * frac_min, 1e-7)
        hi = max(lam_max, lo * 2)
        return np.logspace(np.log10(lo), np.log10(hi), n).tolist()

    def make_groups(n_groups):
        n_groups = max(1, min(n_groups, p))
        return [list(range(j, p, n_groups)) for j in range(n_groups)]

    cfgs = []

    def add(name, cfg, label):
        if name in Solvers:
            cfgs.append((Solvers[name], cfg, label))

    # ── Classical ─────────────────────────────────────────────────────────────
    add('OLSSolver',    {}, 'OLS')
    add('GLMIRLSSolver', {}, 'IRLS')

    # ── Path methods (regression only) ────────────────────────────────────────
    if link == 'identity':
        for lam in lam_grid(50):
            add('LARSSolver', {'lam': float(lam)}, f'LARS lam={lam:.4g}')
        # FusedLasso: 7×8 = 56 configs
        lf1 = lam_grid(7); lf2 = lam_grid(8)
        import itertools
        for l1, l2 in itertools.product(lf1, lf2):
            add('FusedLassoSolver', {'lam1': float(l1), 'lam2': float(l2)},
                f'Fused lam1={l1:.4g} lam2={l2:.4g}')

    # ── Penalized ─────────────────────────────────────────────────────────────
    for lam in lam_grid(50):
        add('RidgeSolver', {'lam': float(lam)}, f'Ridge lam={lam:.4g}')

    for lam in lam_grid(50):
        add('LassoCDSolver', {'lam': float(lam), 'max_iter': 2000}, f'Lasso lam={lam:.4g}')

    for lam in lam_grid(10):
        for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
            add('ElasticNetSolver', {'lam': float(lam), 'alpha': a, 'max_iter': 2000},
                f'EN lam={lam:.4g} a={a}')

    for lam in lam_grid(10):
        for g in [0.5, 1.0, 2.0, 3.0, 5.0]:
            add('AdaptiveLassoSolver', {'lam': float(lam), 'gamma': g},
                f'AdaLasso lam={lam:.4g} g={g}')

    for lam in lam_grid(10):
        for a in [2.5, 3.7, 5.0, 7.0, 10.0]:
            add('SCADLLASolver', {'lam': float(lam), 'a': a, 'max_iter_lasso': 1000},
                f'SCAD lam={lam:.4g} a={a}')

    for lam in lam_grid(10):
        for g in [1.5, 2.0, 3.0, 5.0, 8.0]:
            add('MCPCDSolver', {'lam': float(lam), 'gamma': g},
                f'MCP lam={lam:.4g} g={g}')

    ng_vals = sorted(set([2, max(2, p//10), max(2, p//5), max(2, p//3),
                          min(p, max(5, p//2))]))[:5]
    for lam in lam_grid(10):
        for ng in ng_vals:
            add('GroupLassoSolver', {'lam': float(lam), 'groups': make_groups(ng)},
                f'GrpLasso lam={lam:.4g} G={ng}')

    # ── First-order ───────────────────────────────────────────────────────────
    for lam in lam_grid(50):
        add('ISTASolver',  {'lam': float(lam)}, f'ISTA lam={lam:.4g}')
    for lam in lam_grid(50):
        add('FISTASolver', {'lam': float(lam)}, f'FISTA lam={lam:.4g}')

    # ── Online ────────────────────────────────────────────────────────────────
    for g in [0.001, 0.005, 0.01, 0.05, 0.1]:
        for T in [5, 10, 20, 30, 50, 80, 100, 150, 200, 300]:
            add('SGDSolver', {'gamma0': g, 'n_passes': T}, f'SGD g={g} T={T}')

    for g in [0.001, 0.005, 0.01, 0.05, 0.1]:
        for a in [0.51, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
            add('ImplicitSGDSolver', {'gamma0': g, 'alpha': a, 'n_passes': 10},
                f'ISGD g={g} a={a}')

    for e in [0.01, 0.05, 0.1, 0.5, 1.0]:
        for T in [5, 10, 20, 30, 50, 80, 100, 150, 200, 300]:
            add('AdaGradSolver', {'eta': e, 'n_passes': T}, f'AdaGrad e={e} T={T}')

    for lam in lam_grid(5):
        for g in [0.001, 0.01, 0.05, 0.1, 0.5]:
            for T in [10, 20]:
                add('FOBOSSolver', {'lam': float(lam), 'gamma0': g, 'n_passes': T},
                    f'FOBOS lam={lam:.4g} g={g} T={T}')

    for lam in lam_grid(5):
        for g in [0.001, 0.01, 0.05, 0.1, 0.5]:
            for T in [5, 10]:
                add('RDASolver', {'lam': float(lam), 'gamma0': g, 'n_passes': T},
                    f'RDA lam={lam:.4g} g={g} T={T}')

    for lam in lam_grid(5):
        for g in [0.001, 0.01, 0.05, 0.1, 0.5]:
            for K in [5, 10]:
                add('TruncatedGradientSolver', {'lam': float(lam), 'gamma0': g, 'K': K},
                    f'TruncGrad lam={lam:.4g} g={g} K={K}')

    for nb in [3, 5, 10, 20, 50]:
        for ri in [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.3, 1.0, 3.0, 10.0]:
            add('RenewableGLMSolver', {'n_batches': nb, 'ridge_init': ri},
                f'Renew B={nb} r={ri}')

    return cfgs

# ── Feasibility check ─────────────────────────────────────────────────────────
def is_feasible(sname: str, n: int, p: int, link: str, is_sparse: bool) -> bool:
    """Return False if this solver should be skipped for this dataset."""
    # Regression-only solvers
    if link != 'identity' and sname in REGRESSION_ONLY:
        return False
    # Sparse-incompatible solvers
    if is_sparse and sname in DENSE_ONLY:
        return False
    # Large dense datasets: O(n*p^2) solvers
    if not is_sparse and n * p > MAX_NP_EXPENSIVE and sname in EXPENSIVE_DENSE:
        return False
    # Dense p^3 solvers (ISTA, FISTA, GLMIRLS, RenewableGLM)
    if not is_sparse and p > MAX_P_CUBIC and sname in CUBIC_SOLVERS:
        return False
    # Sparse large-p: dense ops infeasible
    if is_sparse and p > MAX_P_SPARSE_OPS and sname in SPARSE_OPS_SKIP:
        return False
    # SAGA logit on large dense datasets
    if not is_sparse and link == 'logit' and n * p > MAX_NP_SAGA_LOGIT and sname in SAGA_LOGIT:
        return False
    return True

# ── Run one dataset ───────────────────────────────────────────────────────────
_SOLVERS = None  # populated in main()

def run_dataset(ds: dict) -> list[dict]:
    name    = ds['name']
    X, y    = ds['X'], ds['y']
    link    = ds['link']
    n, p    = ds['n'], ds['p']
    is_sp   = ds['is_sparse']

    lam_max = compute_lam_max(X, y)
    print(f"  lam_max = {lam_max:.6g}", flush=True)

    all_cfgs = build_configs(lam_max, p, link)

    records = []
    solver_counts = {}
    for SolverCls, cfg, label in all_cfgs:
        sname = SolverCls.__name__
        if not is_feasible(sname, n, p, link, is_sp):
            continue
        try:
            t0 = time.time()
            result = SolverCls(config=cfg).fit(X, y, link=link)
            elapsed = time.time() - t0
            beta = np.asarray(result.beta_hat, dtype=float).ravel()
            if not np.all(np.isfinite(beta)):
                continue
            records.append(dict(
                dataset=name, cls=sname, label=label,
                beta=beta, bnorm=float(np.linalg.norm(beta)),
                link=link, n=n, p=p, time=elapsed,
                family=FAMILY.get(sname, 'Other'),
            ))
            solver_counts[sname] = solver_counts.get(sname, 0) + 1
        except Exception:
            pass

    print(f"  Solver counts:", flush=True)
    for s, c in sorted(solver_counts.items()):
        print(f"    {s:25s}: {c}", flush=True)
    return records

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _SOLVERS

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None,
                        help='Name of single dataset to run')
    parser.add_argument('--all', action='store_true',
                        help='Run all datasets sequentially')
    parser.add_argument('--list', action='store_true',
                        help='List all dataset names and exit')
    args = parser.parse_args()

    if args.list:
        for name, _, _ in DATASETS:
            print(name)
        return

    # Load solver classes
    print("Loading solvers…", flush=True)
    _SOLVERS = load_solvers()
    print(f"  {len(_SOLVERS)} solvers loaded", flush=True)

    ds_list = DATASETS if args.all else [
        ds for ds in DATASETS if ds[0] == args.dataset
    ]
    if not ds_list:
        print(f"ERROR: dataset '{args.dataset}' not found.")
        sys.exit(1)

    for name, url, ds_type in ds_list:
        out_path = OUT_DIR / f'libsvm_v2_{name}.pkl'
        if out_path.exists():
            print(f"\n[{name}] already done — skipping", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[{name}]", flush=True)
        t_ds = time.time()

        ds = load_dataset(name, url, ds_type)
        if ds is None:
            continue

        print(f"  n={ds['n']}  p={ds['p']}  link={ds['link']}  "
              f"sparse={ds['is_sparse']}  y_std={ds['y_std']:.4g}", flush=True)

        records = run_dataset(ds)
        elapsed = time.time() - t_ds

        # Save per-dataset results (exclude X from metadata to save space)
        ds_meta = {k: v for k, v in ds.items() if k != 'X'}
        with open(out_path, 'wb') as f:
            pickle.dump({'records': records, 'ds_meta': ds_meta}, f)

        print(f"  => {len(records)} runs in {elapsed:.1f}s  saved to {out_path}", flush=True)

    print("\nAll done.", flush=True)


if __name__ == '__main__':
    main()
