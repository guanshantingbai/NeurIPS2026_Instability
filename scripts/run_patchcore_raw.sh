#!/usr/bin/env bash
# Stage 1 — PatchCore raw score generation (GPU + data required). Not invoked by reproduce_*.
# Does not change fast-path defaults elsewhere; only invoked with FULL_RUN=1.
#
# Required:
#   FULL_RUN=1
#   PATCHCORE_DATA_ROOT   — MVTec parent (contains bottle/...) or VisA root per patchcore loaders
#   PATCHCORE_MODELS_RUN  — directory with mvtec_<class>/ or visa_<class>/ PatchCore model folders
#
# Optional:
#   PATCHCORE_DATASET=mvtec|visa   (default: mvtec)
#   PATCHCORE_RAW_OUT              (default: $REPO_ROOT/outputs/cached_results/raw_scores/patchcore)
#   PATCHCORE_GPU=0
#   PATCHCORE_RESUME=1             — pass --resume to score step (append classes; required if CSV exists)
#   PATCHCORE_CLASSES              — comma-separated subset (e.g. bottle for smoke tests)
#   PATCHCORE_EXTRA_ARGS           — extra args passed to run_patchcore_tta_mechanism.py (quoted string)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

if [ "${FULL_RUN:-0}" != "1" ]; then
  echo "ERROR: run_patchcore_raw.sh is full-path only. Set FULL_RUN=1 (and PATCHCORE_DATA_ROOT / PATCHCORE_MODELS_RUN)." >&2
  exit 1
fi

: "${PATCHCORE_DATA_ROOT:?Set PATCHCORE_DATA_ROOT (MVTec or VisA root)}"
: "${PATCHCORE_MODELS_RUN:?Set PATCHCORE_MODELS_RUN (PatchCore models directory)}"

if [ ! -d "$PATCHCORE_DATA_ROOT" ]; then
  echo "ERROR: PATCHCORE_DATA_ROOT is not a directory: $PATCHCORE_DATA_ROOT" >&2
  exit 1
fi
if [ ! -d "$PATCHCORE_MODELS_RUN" ]; then
  echo "ERROR: PATCHCORE_MODELS_RUN is not a directory: $PATCHCORE_MODELS_RUN" >&2
  exit 1
fi

RAW_OUT="${PATCHCORE_RAW_OUT:-$REPO_ROOT/outputs/cached_results/raw_scores/patchcore}"
DATASET="${PATCHCORE_DATASET:-mvtec}"
GPU="${PATCHCORE_GPU:-0}"
mkdir -p "$RAW_OUT"

PY="${PYTHON:-python3}"

SCORE_CMD=(
  "$PY" -m src.models.patchcore_adapter.run_patchcore
  scripts/run_patchcore_tta_mechanism.py
  --step scores
  --out-dir "$RAW_OUT"
  --data-root "$PATCHCORE_DATA_ROOT"
  --models-run "$PATCHCORE_MODELS_RUN"
  --dataset "$DATASET"
  --gpu "$GPU"
  --inference-batch-size "${PATCHCORE_INFERENCE_BATCH_SIZE:-24}"
)

if [ "${PATCHCORE_FAISS_GPU:-0}" = "1" ]; then
  SCORE_CMD+=(--faiss-on-gpu)
fi
if [ "${PATCHCORE_RESUME:-0}" = "1" ]; then
  SCORE_CMD+=(--resume)
fi
if [ -n "${PATCHCORE_CLASSES:-}" ]; then
  SCORE_CMD+=(--classes "$PATCHCORE_CLASSES")
fi
if [ -n "${PATCHCORE_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  SCORE_CMD+=($PATCHCORE_EXTRA_ARGS)
fi

echo "[run_patchcore_raw] scoring -> $RAW_OUT (dataset=$DATASET gpu=$GPU)"
"${SCORE_CMD[@]}"

SCORES_CSV="$RAW_OUT/patchcore_tta_scores.csv"
if [ ! -f "$SCORES_CSV" ]; then
  echo "ERROR: expected scores CSV missing: $SCORES_CSV" >&2
  exit 1
fi

echo "[run_patchcore_raw] exporting unified raw tables"
"$PY" "$REPO_ROOT/src/experiments/app_patchcore_tta/patchcore_export_unified_raw.py" \
  --scores-csv "$SCORES_CSV" \
  --out-dir "$RAW_OUT" \
  --dataset "$DATASET" \
  --models-run "$PATCHCORE_MODELS_RUN" \
  --config-extra "{\"gpu\": ${GPU}}"

for f in unified_raw_scores.csv unified_raw_scores_long.csv; do
  if [ ! -f "$RAW_OUT/$f" ]; then
    echo "ERROR: missing unified export: $RAW_OUT/$f" >&2
    exit 1
  fi
done

echo "[run_patchcore_raw] done — raw scores under $RAW_OUT"
