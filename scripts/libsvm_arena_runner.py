#!/usr/bin/env python3
"""
libsvm_arena_runner.py
======================
Runs on the cluster. Downloads all selected LIBSVM datasets,
processes them, runs all GLM solvers, and saves results.
"""
import os, sys, bz2, gzip, pickle, warnings, time
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

warnings.filterwarnings('ignore')

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

DATA_DIR = Path.home() / 'libsvm_data'
OUT_DIR  = REPO_DIR / 'notebooks'
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ── Dataset catalogue ─────────────────────────────────────────────────────────
BASE_R = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression"
BASE_B = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"

DATASETS = [
    # (name, url, type)
    # ── Regression ──────────────────────────────────────────────────────────
    ("abalone",          f"{BASE_R}/abalone",                          "regression"),
    ("bodyfat",          f"{BASE_R}/bodyfat",                          "regression"),
    ("cadata",           f"{BASE_R}/cadata",                           "regression"),
    ("cpusmall",         f"{BASE_R}/cpusmall",                         "regression"),
    ("eunite2001",       f"{BASE_R}/eunite2001",                       "regression"),
    ("housing",          f"{BASE_R}/housing",                          "regression"),
    ("mg",               f"{BASE_R}/mg",                               "regression"),
    ("mpg",              f"{BASE_R}/mpg",                              "regression"),
    ("pyrim",            f"{BASE_R}/pyrim",                            "regression"),
    ("space_ga",         f"{BASE_R}/space_ga",                         "regression"),
    ("triazines",        f"{BASE_R}/triazines",                        "regression"),
    ("YearPredMSD",      f"{BASE_R}/YearPredictionMSD.bz2",            "regression"),
    ("E2006-tfidf",      f"{BASE_R}/E2006-tfidf/E2006.train.bz2",      "regression"),
    # ── Binary classification ────────────────────────────────────────────────
    ("australian",       f"{BASE_B}/australian",                       "binary"),
    ("breast-cancer",    f"{BASE_B}/breast-cancer",                    "binary"),
    ("cod-rna",          f"{BASE_B}/cod-rna",                          "binary"),
    ("colon-cancer",     f"{BASE_B}/colon-cancer.bz2",                 "binary"),
    ("diabetes",         f"{BASE_B}/diabetes",                         "binary"),
    ("duke",             f"{BASE_B}/duke.bz2",                         "binary"),
    ("fourclass",        f"{BASE_B}/fourclass",                        "binary"),
    ("german-numer",     f"{BASE_B}/german.numer",                     "binary"),
    ("gisette",          f"{BASE_B}/gisette_scale.bz2",                "binary"),
    ("heart",            f"{BASE_B}/heart",                            "binary"),
    ("ijcnn1",           f"{BASE_B}/ijcnn1.bz2",                       "binary"),
    ("ionosphere",       f"{BASE_B}/ionosphere_scale",                 "binary"),
    ("leukemia",         f"{BASE_B}/leu.bz2",                          "binary"),
    ("liver-disorders",  f"{BASE_B}/liver-disorders",                  "binary"),
    ("madelon",          f"{BASE_B}/madelon",                          "binary"),
    ("mushrooms",        f"{BASE_B}/mushrooms",                        "binary"),
    ("news20-binary",    f"{BASE_B}/news20.binary.bz2",                "binary"),
    ("phishing",         f"{BASE_B}/phishing",                         "binary"),
    ("rcv1-binary",      f"{BASE_B}/rcv1_train.binary.bz2",            "binary"),
    ("real-sim",         f"{BASE_B}/real-sim.bz2",                     "binary"),
    ("skin-nonskin",     f"{BASE_B}/skin_nonskin",                     "binary"),
    ("sonar",            f"{BASE_B}/sonar_scale",                      "binary"),
    ("splice",           f"{BASE_B}/splice",                           "binary"),
    ("svmguide1",        f"{BASE_B}/svmguide1",                        "binary"),
    ("svmguide3",        f"{BASE_B}/svmguide3",                        "binary"),
    ("w8a",              f"{BASE_B}/w8a",                              "binary"),
    ("a9a",              f"{BASE_B}/a9a",                              "binary"),
    ("covtype-binary",   f"{BASE_B}/covtype.libsvm.binary.scale.bz2",  "binary"),
    ("epsilon",          f"{BASE_B}/epsilon_normalized.bz2",           "binary"),
]

MAX_N       = 80_000   # subsample rows for large datasets
MAX_P_DENSE = 10_000   # convert to dense if p <= this (SGD solvers need dense rows)
SEED        = 42
rng = np.random.default_rng(SEED)


# ── Download helper ───────────────────────────────────────────────────────────
def download(name: str, url: str) -> Path:
    """Download (and decompress) a dataset, return path to raw text file."""
    final = DATA_DIR / name
    if final.exists() and final.stat().st_size > 200:
        return final
    is_bz = url.endswith('.bz2')
    is_gz  = url.endswith('.gz')
    raw = DATA_DIR / (name + ('.bz2' if is_bz else '.gz' if is_gz else ''))
    if not raw.exists() or raw.stat().st_size < 200:
        print(f"  wget {name} ...", flush=True)
        ret = os.system(f'wget -q -t 5 --timeout=60 -O "{raw}" "{url}"')
        if ret != 0 or not raw.exists() or raw.stat().st_size < 100:
            print(f"  WARN: wget failed for {name}", flush=True)
            return None
    if is_bz:
        with bz2.open(raw, 'rb') as fin, open(final, 'wb') as fout:
            fout.write(fin.read())
        return final
    if is_gz:
        with gzip.open(raw, 'rb') as fin, open(final, 'wb') as fout:
            fout.write(fin.read())
        return final
    return raw  # no compression


# ── Load + preprocess ─────────────────────────────────────────────────────────
def load_dataset(name: str, url: str, ds_type: str):
    path = download(name, url)
    if path is None:
        return None
    try:
        X, y = load_svmlight_file(str(path))
    except Exception as e:
        print(f"  SKIP {name}: {e}", flush=True)
        return None

    n, p = X.shape

    # Binary: remap to {0, 1}
    if ds_type == 'binary':
        labels = np.unique(y)
        if len(labels) != 2:
            print(f"  SKIP {name}: {len(labels)} classes (not binary)", flush=True)
            return None
        y = (y == labels[1]).astype(float)
    else:
        y = y.astype(float)

    # Subsample if too large
    if n > MAX_N:
        idx = rng.choice(n, MAX_N, replace=False)
        X = X[idx]; y = y[idx]; n = MAX_N

    # Decide dense vs sparse
    is_sparse = sp.issparse(X) and p > MAX_P_DENSE
    if sp.issparse(X) and not is_sparse:
        X = X.toarray()

    # Scale
    if is_sparse:
        scaler = MaxAbsScaler()
    else:
        scaler = StandardScaler()
    X = scaler.fit_transform(X)
    if not is_sparse:
        X = np.asarray(X, dtype=np.float64)

    link = 'logit' if ds_type == 'binary' else 'identity'
    return dict(name=name, X=X, y=y, link=link,
                n=int(n), p=int(p), is_sparse=is_sparse,
                source='LIBSVM', type=ds_type)


# ── Solver imports ─────────────────────────────────────────────────────────────
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
    from src.glmzoo.solvers.online.renewable_glm   import RenewableGLMSolver
    return [OLSSolver, GLMIRLSSolver, RidgeSolver, LassoCDSolver, ElasticNetSolver,
            AdaptiveLassoSolver, SCADLLASolver, MCPCDSolver, GroupLassoSolver, LARSSolver,
            ISTASolver, FISTASolver, SGDSolver, ImplicitSGDSolver, AdaGradSolver,
            RDASolver, FOBOSSolver, RenewableGLMSolver]


def build_configs(name: str, p: int) -> list:
    """Return list of config dicts for a solver."""
    lams = np.logspace(-3, 0, 6).tolist()
    # Cap SAGA-based solvers at 500 iterations to keep each config under 30s
    saga_cap = {'max_iter': 500}
    if name == 'OLSSolver':
        return [{}]
    if name == 'GLMIRLSSolver':
        return [{}]
    if name == 'RidgeSolver':
        return [{'lam': l} for l in lams]
    if name == 'LassoCDSolver':
        return [{**{'lam': l}, **saga_cap} for l in lams]
    if name in ('ISTASolver', 'FISTASolver'):
        return [{'lam': l} for l in lams]
    if name == 'ElasticNetSolver':
        return [{**{'lam': l, 'alpha': a}, **saga_cap} for l in lams for a in [0.1, 0.5, 0.9]]
    if name in ('AdaptiveLassoSolver', 'SCADLLASolver', 'MCPCDSolver'):
        # Cap inner Lasso iterations to avoid extremely long runs on large-n datasets
        lasso_cap = {'max_iter_lasso': 500}
        return [{**{'lam': l}, **lasso_cap} for l in lams]
    if name == 'GroupLassoSolver':
        g = max(2, p // 10)
        return [{'lam': l, 'n_groups': g} for l in lams]
    if name == 'LARSSolver':
        steps = sorted(set([min(p, k) for k in [10, 20, 50, 100, p]]))
        return [{'max_iter': k} for k in steps]
    if name in ('SGDSolver', 'ImplicitSGDSolver', 'AdaGradSolver', 'RDASolver', 'FOBOSSolver'):
        return [{'lr': lr} for lr in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]]
    if name == 'RenewableGLMSolver':
        return [{}]
    return [{}]


# ── Solver family mapping ─────────────────────────────────────────────────────
FAMILY = {
    'OLSSolver': 'Classical', 'GLMIRLSSolver': 'Classical',
    'RidgeSolver': 'Penalized', 'LassoCDSolver': 'Penalized', 'ElasticNetSolver': 'Penalized',
    'AdaptiveLassoSolver': 'Penalized', 'SCADLLASolver': 'Penalized', 'MCPCDSolver': 'Penalized',
    'GroupLassoSolver': 'Penalized', 'LARSSolver': 'Penalized',
    'ISTASolver': 'First-order', 'FISTASolver': 'First-order',
    'SGDSolver': 'Online', 'ImplicitSGDSolver': 'Online', 'AdaGradSolver': 'Online',
    'RDASolver': 'Online', 'FOBOSSolver': 'Online', 'RenewableGLMSolver': 'Online',
}

# Solvers that can't handle sparse matrices
DENSE_ONLY = {'LARSSolver', 'GroupLassoSolver', 'SCADLLASolver', 'MCPCDSolver', 'AdaptiveLassoSolver'}

# Solvers whose X'X / kernel matrix is infeasible for large sparse p
# (now defined alongside EIGVALSH_SOLVERS below)
MAX_P_FOR_DENSE_OPS = 10_000  # skip these on sparse data with p > this

# LARS path (O(n*p^2)) and GroupLasso are infeasible for large dense datasets
# SCAD/MCP also run many Lasso iters internally (50 outer × Lasso) — too slow for large n*p
# Skip when n*p > 5M (e.g. gisette 6000×5000=30M; leukemia 38×7129=270k OK)
MAX_NP_FOR_PATH = 5_000_000
PATH_EXPENSIVE = {'LARSSolver', 'GroupLassoSolver', 'SCADLLASolver', 'MCPCDSolver'}

# ISTA/FISTA compute eigvalsh(X'X) which is O(p^3) — infeasible for p > 2000
# GLMIRLSSolver Newton step is also O(np^2) for large p
# RenewableGLM computes X_b.T @ diag(V_b) @ X_b — a (p×p) matrix per batch
MAX_P_FOR_EIGVALSH = 2_000   # skip for dense datasets with p > this
EIGVALSH_SOLVERS = {'ISTASolver', 'FISTASolver', 'GLMIRLSSolver', 'RenewableGLMSolver'}

# Same solvers also infeasible for sparse data with large p
LARGE_SPARSE_SKIP = {'ISTASolver', 'FISTASolver', 'RidgeSolver', 'GLMIRLSSolver', 'RenewableGLMSolver'}

# LassoCD/ElasticNet SAGA for logit on large datasets is very slow (n*p ops per epoch × 500 epochs)
# Skip when n*p > 5M for logit link (same threshold as PATH_EXPENSIVE)
MAX_NP_FOR_SAGA_LOGIT = 5_000_000
SAGA_LOGIT_SOLVERS = {'LassoCDSolver', 'ElasticNetSolver'}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70, flush=True)
    print(f"LIBSVM Arena Runner — {len(DATASETS)} datasets", flush=True)
    print("=" * 70, flush=True)

    # Check for existing partial results
    cache_path = OUT_DIR / 'libsvm_arena_results.pkl'
    done_datasets = set()
    records = []
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        records = cached['df_all'].to_dict('records')
        done_datasets = set(cached['df_all']['dataset'].unique())
        print(f"Resuming: {len(done_datasets)} datasets already done.", flush=True)

    # Load datasets
    print("\n=== Loading datasets ===", flush=True)
    loaded = {}
    for name, url, ds_type in DATASETS:
        t0 = time.time()
        ds = load_dataset(name, url, ds_type)
        if ds:
            loaded[name] = ds
            print(f"  OK  {name:25s}  n={ds['n']:6d}  p={ds['p']:8d}  "
                  f"sparse={ds['is_sparse']}  ({time.time()-t0:.1f}s)", flush=True)
        else:
            print(f"  FAIL {name}", flush=True)
    print(f"\nLoaded {len(loaded)} / {len(DATASETS)} datasets", flush=True)

    # Load solvers
    print("\n=== Loading solvers ===", flush=True)
    try:
        Solvers = load_solvers()
        print(f"Loaded {len(Solvers)} solvers", flush=True)
    except Exception as e:
        print(f"ERROR loading solvers: {e}", flush=True)
        sys.exit(1)

    # Run arena
    print("\n=== Running arena ===", flush=True)
    t_total = time.time()

    for ds_name, ds in loaded.items():
        if ds_name in done_datasets:
            print(f"  SKIP {ds_name} (cached)", flush=True)
            continue

        X, y, link = ds['X'], ds['y'], ds['link']
        n, p = ds['n'], ds['p']
        is_sparse = ds['is_sparse']
        print(f"\n[{ds_name}]  n={n}  p={p}  link={link}  sparse={is_sparse}", flush=True)

        ds_records = []
        for SolverCls in Solvers:
            sname = SolverCls.__name__

            # Compatibility filters
            if is_sparse and sname in DENSE_ONLY:
                continue
            if sname == 'OLSSolver' and link != 'identity':
                continue
            # SAGA-based logit solvers can't be preempted; skip for sparse
            if is_sparse and link == 'logit' and sname in ('LassoCDSolver', 'ElasticNetSolver'):
                continue
            # Dense matrix ops (X'X, kernel, column_stack) infeasible for large sparse p
            if is_sparse and p > MAX_P_FOR_DENSE_OPS and sname in LARGE_SPARSE_SKIP:
                continue
            # LARS path (O(n*p^2)) and GroupLasso infeasible for large dense datasets
            if not is_sparse and n * p > MAX_NP_FOR_PATH and sname in PATH_EXPENSIVE:
                continue
            # ISTA/FISTA compute eigvalsh(X'X) O(p^3); GLMIRLS Newton O(np^2) — too slow for large dense p
            if not is_sparse and p > MAX_P_FOR_EIGVALSH and sname in EIGVALSH_SOLVERS:
                continue
            # SAGA logit on large dense datasets: O(n*p) per iteration × 500 iters × 6 configs
            if not is_sparse and link == 'logit' and n * p > MAX_NP_FOR_SAGA_LOGIT and sname in SAGA_LOGIT_SOLVERS:
                continue

            configs = build_configs(sname, p)
            n_ok = 0
            for cfg in configs:
                try:
                    t1 = time.time()
                    solver = SolverCls(config=cfg)
                    result = solver.fit(X, y, link=link)
                    elapsed = time.time() - t1
                    beta = np.asarray(result.beta_hat, dtype=float).ravel()
                    lbl = ','.join(f'{k}={v:.3g}' for k, v in cfg.items()) if cfg else 'default'
                    ds_records.append(dict(
                        dataset=ds_name, cls=sname, label=lbl,
                        beta=beta, bnorm=float(np.linalg.norm(beta)),
                        link=link, n=n, p=p, time=elapsed,
                        family=FAMILY.get(sname, 'Other'),
                    ))
                    n_ok += 1
                except Exception:
                    pass
            if n_ok:
                print(f"    {sname:20s}  {n_ok}/{len(configs)} configs ok", flush=True)

        records.extend(ds_records)
        print(f"  {ds_name}: {len(ds_records)} runs", flush=True)

        # Save partial results after each dataset
        import pandas as pd
        df_partial = pd.DataFrame(records)
        with open(cache_path, 'wb') as f:
            pickle.dump({'df_all': df_partial,
                         'datasets': {k: {kk: vv for kk, vv in v.items() if kk != 'X'}
                                      for k, v in loaded.items()}}, f)

    import pandas as pd
    df = pd.DataFrame(records)
    total_time = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"Total: {len(records)} runs  |  {len(df['dataset'].unique())} datasets  |  {total_time:.1f}s", flush=True)

    # Final save (include loaded dict without X for metadata)
    with open(cache_path, 'wb') as f:
        pickle.dump({'df_all': df,
                     'datasets': {k: {kk: vv for kk, vv in v.items() if kk != 'X'}
                                  for k, v in loaded.items()}}, f)
    # Separate datasets file (with X) for figure generation
    with open(OUT_DIR / 'libsvm_datasets.pkl', 'wb') as f:
        pickle.dump(loaded, f)

    print(f"Saved to {cache_path}", flush=True)
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
