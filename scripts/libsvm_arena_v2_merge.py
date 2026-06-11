"""
libsvm_arena_v2_merge.py
========================
Merge all per-dataset pkl files (produced by libsvm_arena_v2.py) into
a single result file that matches the format expected by
generate_libsvm_plots.py.

Output:
    notebooks/libsvm_arena_results.pkl  (same file used by the plot generator)

Usage:
    python scripts/libsvm_arena_v2_merge.py
    python scripts/libsvm_arena_v2_merge.py --dry-run
"""
from __future__ import annotations
import sys, argparse, pickle
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
V2_DIR   = BASE_DIR / 'notebooks' / 'libsvm_v2'
OUT_PATH = BASE_DIR / 'notebooks' / 'libsvm_arena_results.pkl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Show summary without saving')
    args = parser.parse_args()

    pkl_files = sorted(V2_DIR.glob('libsvm_v2_*.pkl'))
    if not pkl_files:
        print(f"ERROR: no .pkl files found in {V2_DIR}")
        sys.exit(1)

    all_records: list[dict] = []
    datasets_meta: dict     = {}
    failed: list[str]       = []

    for fp in pkl_files:
        name = fp.stem.replace('libsvm_v2_', '')
        try:
            with open(fp, 'rb') as f:
                obj = pickle.load(f)
            records   = obj['records']
            ds_meta   = obj['ds_meta']
            all_records.extend(records)
            datasets_meta[name] = ds_meta
            print(f"  OK  {name:25s}: {len(records):5d} runs  "
                  f"n={ds_meta['n']:6d}  p={ds_meta['p']:6d}  "
                  f"link={ds_meta['link']}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed.append(name)

    if failed:
        print(f"\nWARN: {len(failed)} datasets failed to load: {failed}")

    print(f"\nTotal: {len(all_records)} runs across {len(datasets_meta)} datasets")

    if args.dry_run:
        print("(dry-run — not saving)")
        return

    # Build DataFrame (same schema as original arena)
    df = pd.DataFrame([
        {
            'dataset': r['dataset'],
            'cls':     r['cls'],
            'label':   r['label'],
            'beta':    np.asarray(r['beta'], dtype=float),
            'bnorm':   float(r['bnorm']),
            'link':    r['link'],
            'n':       int(r['n']),
            'p':       int(r['p']),
            'time':    float(r['time']),
            'family':  r.get('family', 'Other'),
        }
        for r in all_records
    ])

    out = {'df_all': df, 'datasets': datasets_meta}
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(out, f)

    print(f"\nSaved → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
