#!/usr/bin/env bash
# PatchCore full-path raw score generation (GPU + data required).
# Does not change fast-path defaults elsewhere; invoke explicitly or via FULL_RUN=1 on appendix F.
#
# Required:
#   PATCHCORE_DATA_ROOT   — MVTec parent (contains bottle/...) or VisA root per patchcore datasets
#   PATCHCORE_MODELS_RUN  — directory with mvtec_<class>/ or visa_<class>/ PatchCore model folders
#
# Optional:
#   PATCHCORE_DATASET=mvtec|visa   (default: mvtec)
#   PATCHCORE_RAW_OUT              (default: $REPO_ROOT/outputs/cached_results/raw_scores/patchcore)
#   PATCHCORE_GPU=0
#   PATCHCORE_RESUME=1             — pass --resume to score step
#   PATCHCORE_EXTRA_ARGS           — extra args passed to run_patchcore_tta_mechanism.py (quoted string)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

: "${PATCHCORE_DATA_ROOT:?Set PATCHCORE_DATA_ROOT (MVTec or VisA root)}"
: "${PATCHCORE_MODELS_RUN:?Set PATCHCORE_MODELS_RUN (PatchCore models directory)}"

RAW_OUT="${PATCHCORE_RAW_OUT:-$REPO_ROOT/outputs/cached_results/raw_scores/patchcore}"
DATASET="${PATCHCORE_DATASET:-mvtec}"
GPU="${PATCHCORE_GPU:-0}"
mkdir -p "$RAW_OUT"

SCORE_ARGS=(python -m src.models.patchcore_adapter.run_patchcore "scripts/run_patchcore_tta_mechanism.py")
SCORE_ARGS+=(--step scores)
SCORE_ARGS+=(--out-dir "$RAW_OUT")
SCORE_ARGS+=(--data-root "$PATCHCORE_DATA_ROOT")
SCORE_ARGS+=(--models-run "$PATCHCORE_MODELS_RUN")
SCORE_ARGS+=(--dataset "$DATASET")
SCORE_ARGS+=(--gpu "$GPU")
SCORE_ARGS+=(--inference-batch-size "${PATCHCORE_INFERENCE_BATCH_SIZE:-24}")

if [ "${PATCHCORE_FAISS_GPU:-0}" = "1" ]; then
  SCORE_ARGS+=(--faiss-on-gpu)
fi
if [ "${PATCHCORE_RESUME:-0}" = "1" ]; then
  SCORE_ARGS+=(--resume)
fi
# shellcheck disable=SC2206
# Optional: space-separated extra CLI tokens (e.g. --max-classes 2). Avoid spaces inside paths.
if [ -n "${PATCHCORE_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206,SC2086
  SCORE_ARGS+=($PATCHCORE_EXTRA_ARGS)
fi

echo "[run_patchcore_raw] scoring -> $RAW_OUT"
"${SCORE_ARGS[@]}"

SCORES_CSV="$RAW_OUT/patchcore_tta_scores.csv"
if [ ! -f "$SCORES_CSV" ]; then
  echo "ERROR: expected scores CSV missing: $SCORES_CSV" >&2
  exit 1
fi

echo "[run_patchcore_raw] exporting unified raw tables"
python "$REPO_ROOT/src/experiments/app_patchcore_tta/patchcore_export_unified_raw.py" \
  --scores-csv "$SCORES_CSV" \
  --out-dir "$RAW_OUT" \
  --dataset "$DATASET" \
  --models-run "$PATCHCORE_MODELS_RUN" \
  --config-extra "{\"gpu\": ${GPU}}"

echo "[run_patchcore_raw] done"
