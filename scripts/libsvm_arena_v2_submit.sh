#!/usr/bin/env bash
# libsvm_arena_v2_submit.sh
# =============================================================================
# Distributed submission of libsvm_arena_v2.py across the cluster.
# Each dataset runs as an independent background process.
# Concurrency is capped to avoid saturating RAM (128 cores / 252 GB).
#
# Usage:
#   bash scripts/libsvm_arena_v2_submit.sh           # run all
#   bash scripts/libsvm_arena_v2_submit.sh abalone   # run one
#   JOBS=8 bash scripts/libsvm_arena_v2_submit.sh    # custom concurrency
#
# On the cluster:
#   cd /home/23038969r/regression-zoo
#   bash scripts/libsvm_arena_v2_submit.sh 2>&1 | tee libsvm_v2_run.log
# =============================================================================

set -euo pipefail

SCRIPT="scripts/libsvm_arena_v2.py"
LOGDIR="notebooks/libsvm_v2/logs"
JOBS="${JOBS:-16}"   # max parallel datasets; 16 is safe on 128-core / 252GB

mkdir -p "$LOGDIR"

# ── Build list of datasets to run ────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    mapfile -t DATASETS < <(python3 "$SCRIPT" --list)
fi

echo "===== LIBSVM Arena v2 ====="
echo "Datasets to run: ${#DATASETS[@]}"
echo "Max parallel jobs: $JOBS"
echo "==============================="

active=0
declare -A PIDS   # dataset → PID

launch() {
    local name="$1"
    local logfile="$LOGDIR/${name}.log"
    echo "[launch] $name  (log → $logfile)"
    python3 "$SCRIPT" --dataset "$name" > "$logfile" 2>&1 &
    PIDS[$name]=$!
    active=$(( active + 1 ))
}

for name in "${DATASETS[@]}"; do
    # Skip if already done
    if [[ -f "notebooks/libsvm_v2/libsvm_v2_${name}.pkl" ]]; then
        echo "[skip] $name  (already done)"
        continue
    fi

    # Wait if we're at max concurrency
    while (( active >= JOBS )); do
        wait -n 2>/dev/null || true
        # recount active
        active=0
        for n in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$n]}" 2>/dev/null; then
                active=$(( active + 1 ))
            else
                unset "PIDS[$n]"
            fi
        done
    done

    launch "$name"
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining jobs…"
wait

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "===== Summary ====="
success=0; failed=0
for name in "${DATASETS[@]}"; do
    pkl="notebooks/libsvm_v2/libsvm_v2_${name}.pkl"
    if [[ -f "$pkl" ]]; then
        echo "  OK  $name"
        success=$(( success + 1 ))
    else
        echo "  FAIL $name  (check $LOGDIR/${name}.log)"
        failed=$(( failed + 1 ))
    fi
done
echo "Done: $success / $(( success + failed ))"
