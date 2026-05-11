#!/usr/bin/env bash
# Retrain PromptAD CLS for every (dataset, class, k-shot) using train_cls.py.
#
# Training-time image-level AUROC (for checkpoint selection) follows
# utils.instability_penalty.metric_cal_img_with_instability_penalty:
#   - --instability-penalty-lambda 0  -> same as original harmonic fusion AUROC
#   - --instability-penalty-lambda >0 -> optimize checkpoint for penalized score
#
# Usage:
#   bash PromptAD/bash/run_cls_train_all_settings.sh
#
# Instability-aware retrain (example λ=0.1):
#   INSTABILITY_PENALTY_LAMBDA=0.1 bash PromptAD/bash/run_cls_train_all_settings.sh
#
# Environment:
#   ROOT_DIR                    - default: <repo>/result_offline_train (separate from infer output)
#   PYTHON                      - default: python
#   SEED                        - default: 111
#   GPU_ID                      - default: 0
#   DATASETS                    - default: mvtec (space-separated; add visa if needed)
#   SHOTS                       - default: 1 2 4
#   EVAL_FREQ                   - default: 2 (same as run_cls.py)
#   INSTABILITY_PENALTY_LAMBDA  - optional; if set, passed to train_cls (e.g. 0.1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/result_offline_train}"
PYTHON_BIN="${PYTHON:-python}"
SEED="${SEED:-111}"
GPU_ID="${GPU_ID:-0}"
DATASETS="${DATASETS:-mvtec}"
SHOTS="${SHOTS:-1 2 4}"
EVAL_FREQ="${EVAL_FREQ:-2}"

export DATASETS
export SHOTS

mkdir -p "${ROOT_DIR}"

EXTRA_ARGS=()
if [[ -n "${INSTABILITY_PENALTY_LAMBDA:-}" ]]; then
  EXTRA_ARGS+=(--instability-penalty-lambda "${INSTABILITY_PENALTY_LAMBDA}")
fi

echo "[INFO] ROOT_DIR=${ROOT_DIR}  (checkpoints + csv + run_records for training)"
echo "[INFO] SEED=${SEED} GPU_ID=${GPU_ID} EVAL_FREQ=${EVAL_FREQ}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "[INFO] Extra train flags: ${EXTRA_ARGS[*]}"
else
  echo "[INFO] Instability penalty lambda: 0 (harmonic AUROC only; set INSTABILITY_PENALTY_LAMBDA to enable)"
fi

plan="$(
  "${PYTHON_BIN}" - <<'PY'
import os
from datasets import dataset_classes

datasets = os.environ["DATASETS"].split()
shots = os.environ["SHOTS"].split()
for ds in datasets:
    if ds not in dataset_classes:
        raise SystemExit(f"unknown dataset in DATASETS: {ds!r}")
    for shot in shots:
        for cls in dataset_classes[ds]:
            print(f"{ds}\t{cls}\t{shot}")
PY
)"

ok=0
fail=0

while IFS=$'\t' read -r ds cls shot; do
  [[ -z "${ds:-}" ]] && continue

  echo "[RUN] train dataset=${ds} class=${cls} k-shot=${shot} seed=${SEED}"
  if "${PYTHON_BIN}" train_cls.py \
    --dataset "${ds}" \
    --class_name "${cls}" \
    --k-shot "${shot}" \
    --root-dir "${ROOT_DIR}" \
    --seed "${SEED}" \
    --gpu-id "${GPU_ID}" \
    --eval-freq "${EVAL_FREQ}" \
    "${EXTRA_ARGS[@]}"; then
    ok=$((ok + 1))
  else
    echo "[FAIL] dataset=${ds} class=${cls} k-shot=${shot}" >&2
    fail=$((fail + 1))
  fi
done <<< "${plan}"

echo "[DONE] ok=${ok} fail=${fail}"
if [[ "${fail}" -gt 0 ]]; then
  exit 2
fi
