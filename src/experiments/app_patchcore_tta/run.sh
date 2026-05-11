#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

mkdir -p outputs/figures/app_patchcore_tta outputs/cached_results/app_patchcore_tta

SRC_DIR="$REPO_ROOT/result_analysis/patchcore_tta"
if [ ! -d "$SRC_DIR" ]; then
  echo "ERROR: Appendix F fast path requires $SRC_DIR (cached lightweight CSV/figures)." >&2
  echo "       Full rerun is not default; set FULL_RUN=1 and export PATCHCORE_DATA_ROOT and PATCHCORE_MODELS_RUN." >&2
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

if [ "${FULL_RUN:-0}" = "1" ]; then
  if [ -z "${PATCHCORE_DATA_ROOT:-}" ] || [ -z "${PATCHCORE_MODELS_RUN:-}" ]; then
    echo "ERROR: FULL_RUN=1 requires PATCHCORE_DATA_ROOT and PATCHCORE_MODELS_RUN." >&2
    exit 1
  fi
  echo "WARNING: FULL_RUN=1 invokes PatchCore TTA mechanism (GPU / data)." >&2
  python -m src.models.patchcore_adapter.run_patchcore "scripts/run_patchcore_tta_mechanism.py" \
    --data-root "$PATCHCORE_DATA_ROOT" \
    --models-run "$PATCHCORE_MODELS_RUN" \
    --out-dir "$REPO_ROOT/outputs/cached_results/app_patchcore_tta_full"
fi

echo "app_patchcore_tta: fast path (cached-only copy from result_analysis/patchcore_tta) OK"
