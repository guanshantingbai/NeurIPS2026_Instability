#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

mkdir -p outputs/figures/app_patchcore_tta outputs/cached_results/app_patchcore_tta

RAW_SCORES_DIR="$REPO_ROOT/outputs/cached_results/raw_scores/patchcore"
RAW_SCORES_CSV="$RAW_SCORES_DIR/patchcore_tta_scores.csv"

if [ "${FULL_RUN:-0}" = "1" ]; then
  if [ -z "${PATCHCORE_DATA_ROOT:-}" ] || [ -z "${PATCHCORE_MODELS_RUN:-}" ]; then
    echo "ERROR: FULL_RUN=1 requires PATCHCORE_DATA_ROOT and PATCHCORE_MODELS_RUN." >&2
    exit 1
  fi
  echo "WARNING: FULL_RUN=1 runs PatchCore raw scoring + unified export + analyze (GPU / data)." >&2
  bash "$REPO_ROOT/scripts/run_patchcore_raw.sh"

  if [ ! -f "$RAW_SCORES_CSV" ]; then
    echo "ERROR: missing raw scores after run_patchcore_raw: $RAW_SCORES_CSV" >&2
    exit 1
  fi

  OUT_ANALYSIS="$REPO_ROOT/outputs/cached_results/app_patchcore_tta"
  mkdir -p "$OUT_ANALYSIS" "$REPO_ROOT/outputs/figures/app_patchcore_tta"

  python -m src.models.patchcore_adapter.run_patchcore "scripts/run_patchcore_tta_mechanism.py" \
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

  echo "app_patchcore_tta: FULL_RUN analysis outputs under $OUT_ANALYSIS and figures copied to outputs/figures/app_patchcore_tta/"
  exit 0
fi

SRC_DIR="$REPO_ROOT/result_analysis/patchcore_tta"
if [ ! -d "$SRC_DIR" ]; then
  echo "ERROR: Appendix F fast path requires $SRC_DIR (cached lightweight CSV/figures)." >&2
  echo "       For model-level rerun: FULL_RUN=1 with PATCHCORE_DATA_ROOT and PATCHCORE_MODELS_RUN." >&2
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
