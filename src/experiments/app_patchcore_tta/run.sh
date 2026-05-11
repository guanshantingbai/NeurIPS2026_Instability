#!/usr/bin/env bash
# Stage 2 only: never runs scripts/run_patchcore_raw.sh (Stage 1).
# - If raw evidence exists (unified_raw_scores_long.csv), run analyze from patchcore_tta_scores.csv.
# - Else copy bundled fast-path assets from result_analysis/patchcore_tta/.
# - PATCHCORE_FROM_RAW=1 requires raw evidence; if missing, exit with hint to run Stage 1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

mkdir -p outputs/figures/app_patchcore_tta outputs/cached_results/app_patchcore_tta

RAW_SCORES_DIR="$REPO_ROOT/outputs/cached_results/raw_scores/patchcore"
RAW_SCORES_CSV="$RAW_SCORES_DIR/patchcore_tta_scores.csv"
UNIFIED_LONG="$RAW_SCORES_DIR/unified_raw_scores_long.csv"

if [ -f "$UNIFIED_LONG" ]; then
  if [ ! -f "$RAW_SCORES_CSV" ]; then
    echo "ERROR: found $UNIFIED_LONG but missing $RAW_SCORES_CSV (inconsistent Stage 1 output)." >&2
    echo "       Re-run Stage 1: FULL_RUN=1 bash scripts/run_patchcore_raw.sh" >&2
    exit 1
  fi
  if [ -z "${PATCHCORE_DATA_ROOT:-}" ] || [ -z "${PATCHCORE_MODELS_RUN:-}" ]; then
    echo "ERROR: raw-derived Appendix F needs PATCHCORE_DATA_ROOT and PATCHCORE_MODELS_RUN (required by upstream analyze argparse)." >&2
    exit 1
  fi
  echo "app_patchcore_tta: Stage 2 from cached raw evidence (analyze only, no scoring)."
  OUT_ANALYSIS="$REPO_ROOT/outputs/cached_results/app_patchcore_tta"
  mkdir -p "$OUT_ANALYSIS" "$REPO_ROOT/outputs/figures/app_patchcore_tta"

  PY="${PYTHON:-python3}"
  "$PY" -m src.models.patchcore_adapter.run_patchcore scripts/run_patchcore_tta_mechanism.py \
    --step analyze \
    --scores-csv "$RAW_SCORES_CSV" \
    --out-dir "$OUT_ANALYSIS" \
    --data-root "$PATCHCORE_DATA_ROOT" \
    --models-run "$PATCHCORE_MODELS_RUN" \
    --dataset "${PATCHCORE_DATASET:-mvtec}" \
    --gpu "${PATCHCORE_GPU:-0}"

  shopt -s nullglob
  for f in "$OUT_ANALYSIS"/*.png "$OUT_ANALYSIS"/*.pdf; do
    cp -f "$f" "$REPO_ROOT/outputs/figures/app_patchcore_tta/"
  done

  for f in unified_raw_scores.csv unified_raw_scores_long.csv; do
    if [ -f "$RAW_SCORES_DIR/$f" ]; then
      cp -f "$RAW_SCORES_DIR/$f" "$OUT_ANALYSIS/"
    fi
  done

  echo "app_patchcore_tta: Stage 2 OK (raw evidence -> analyze -> outputs)."
  exit 0
fi

if [ "${PATCHCORE_FROM_RAW:-0}" = "1" ]; then
  echo "ERROR: PATCHCORE_FROM_RAW=1 but raw evidence missing: $UNIFIED_LONG" >&2
  echo "       Run Stage 1 first: FULL_RUN=1 bash scripts/run_patchcore_raw.sh" >&2
  echo "       (with PATCHCORE_DATA_ROOT, PATCHCORE_MODELS_RUN, etc.; see docs/FULLPATH_PATCHCORE.md)" >&2
  exit 1
fi

SRC_DIR="$REPO_ROOT/result_analysis/patchcore_tta"
if [ ! -d "$SRC_DIR" ]; then
  echo "ERROR: Appendix F fast path requires $SRC_DIR (cached lightweight CSV/figures)." >&2
  echo "       For Stage 1 raw extraction: FULL_RUN=1 bash scripts/run_patchcore_raw.sh" >&2
  echo "       Then Stage 2 will pick up outputs/cached_results/raw_scores/patchcore/unified_raw_scores_long.csv" >&2
  exit 1
fi

shopt -s nullglob
csv_files=("$SRC_DIR"/*.csv)
png_files=("$SRC_DIR"/*.png)
pdf_files=("$SRC_DIR"/*.pdf)
if [ "${#csv_files[@]}" -eq 0 ]; then
  echo "ERROR: no CSV files under $SRC_DIR — cannot populate cached_results." >&2
  exit 1
fi
cp -f "${csv_files[@]}" outputs/cached_results/app_patchcore_tta/
if [ "${#png_files[@]}" -gt 0 ]; then
  cp -f "${png_files[@]}" outputs/figures/app_patchcore_tta/
fi
if [ "${#pdf_files[@]}" -gt 0 ]; then
  cp -f "${pdf_files[@]}" outputs/figures/app_patchcore_tta/
fi

echo "app_patchcore_tta: fast path (cached-only copy from result_analysis/patchcore_tta) OK"
