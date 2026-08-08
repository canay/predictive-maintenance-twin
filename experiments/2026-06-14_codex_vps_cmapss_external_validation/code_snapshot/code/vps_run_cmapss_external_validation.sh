#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${FH8_RUN_ID:-cmapss_external_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

export FH8_RUN_ID="$RUN_ID"
export FH8_N_JOBS="${FH8_N_JOBS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pyc}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

notify() {
  if command -v mesaj >/dev/null 2>&1; then
    mesaj "$1" || true
  fi
}

on_error() {
  rc=$?
  notify "fh8 C-MAPSS external validation ERROR run_id=$RUN_ID rc=$rc"
  exit "$rc"
}
trap on_error ERR

cd "$BASE_DIR"
notify "fh8 C-MAPSS external validation START run_id=$RUN_ID datasets=FD001-FD004 models=constant/ridge/hgb/rf"

"$PYTHON_BIN" code/run_cmapss_external_validation.py \
  --run-id "$RUN_ID" \
  --datasets FD001 FD002 FD003 FD004 \
  --models constant ridge hgb rf \
  --seeds 0 1 2 \
  --n-jobs "$FH8_N_JOBS" \
  2>&1 | tee "$LOG_DIR/${RUN_ID}_run.log"

"$PYTHON_BIN" code/plot_cmapss_external_validation.py \
  --run-id "$RUN_ID" \
  2>&1 | tee "$LOG_DIR/${RUN_ID}_plot.log"

notify "fh8 C-MAPSS external validation DONE run_id=$RUN_ID"
