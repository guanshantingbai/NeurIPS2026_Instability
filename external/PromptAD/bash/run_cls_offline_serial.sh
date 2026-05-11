#!/usr/bin/env bash
# Offline-friendly serial job: full-setting inference first, then full-setting retraining.
# Designed for ``nohup`` so SSH disconnect does not kill the run.
#
# Default directory split (no collisions):
#   CHECKPOINT_ROOT   - read existing .pt (default: <repo>/result_round1)
#   OUTPUT_ROOT       - inference csv/imgs (default: <repo>/result_offline_infer)
#   TRAIN_ROOT_DIR    - new training outputs (default: <repo>/result_offline_train)
#
# One-shot (detach, log to file, survive SSH drop):
#
#   cd /path/to/PromptAD
#   mkdir -p logs
#   nohup env GPU_ID=0 bash bash/run_cls_offline_serial.sh > logs/offline_serial.log 2>&1 &
#   echo $! > logs/offline_serial.pid
#   disown
#
# Tail progress:
#   tail -f logs/offline_serial.log
#
# Optional env (same as the individual scripts):
#   CHECKPOINT_ROOT, OUTPUT_ROOT, TRAIN_ROOT_DIR, SEED, DATASETS, SHOTS, GPU_ID,
#   VIS, SKIP_MISSING, EVAL_FREQ, INSTABILITY_PENALTY_LAMBDA, PYTHON
#
# Alternative: GNU screen / tmux session instead of nohup (also survives SSH).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export REPO_ROOT

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/result_round1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/result_offline_infer}"
TRAIN_ROOT_DIR="${TRAIN_ROOT_DIR:-${REPO_ROOT}/result_offline_train}"

# Stale ROOT_DIR in the environment would trigger legacy mode in run_cls_infer_all_settings.sh
# and force checkpoint + csv into the same tree; drop it so CHECKPOINT_ROOT/OUTPUT_ROOT apply.
unset ROOT_DIR 2>/dev/null || true

echo "=========================================="
echo "[SERIAL] $(date -Iseconds)"
echo "CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_ROOT_DIR=${TRAIN_ROOT_DIR}"
echo "=========================================="

export CHECKPOINT_ROOT
export OUTPUT_ROOT

echo ""
echo "[PHASE 1/2] Inference (all settings) -> OUTPUT_ROOT"
bash "${SCRIPT_DIR}/run_cls_infer_all_settings.sh"

export ROOT_DIR="${TRAIN_ROOT_DIR}"
unset CHECKPOINT_ROOT OUTPUT_ROOT

echo ""
echo "[PHASE 2/2] Retraining (all settings) -> TRAIN_ROOT_DIR=${TRAIN_ROOT_DIR}"
bash "${SCRIPT_DIR}/run_cls_train_all_settings.sh"

echo ""
echo "[SERIAL] Done $(date -Iseconds)"
