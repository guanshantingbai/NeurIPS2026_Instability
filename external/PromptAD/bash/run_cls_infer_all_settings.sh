#!/usr/bin/env bash
# Run PromptAD image-level inference (test_cls.py) for every setting: dataset × k-shot × class.
#
# Checkpoints are read from CHECKPOINT_ROOT; csv/imgs go to OUTPUT_ROOT (defaults differ so
# inference does not overwrite a separate retraining tree). Legacy: set OUTPUT_ROOT empty to
# use CHECKPOINT_ROOT for everything (same as old ROOT_DIR-only behavior).
#
# Writes under:
#   ${OUTPUT_ROOT}/{dataset}/k_{k}/csv/   (metrics + per_sample + per_sample_instability)
#   ${OUTPUT_ROOT}/{dataset}/k_{k}/imgs/  (if --vis true)
#
# Checkpoints:
#   ${CHECKPOINT_ROOT}/{dataset}/k_{k}/checkpoint/CLS-Seed_${SEED}-${class}-check_point.pt
#
# Usage (from anywhere):
#   bash PromptAD/bash/run_cls_infer_all_settings.sh
#
# Common overrides:
#   CHECKPOINT_ROOT=/path/to/result_round1 OUTPUT_ROOT=/path/to/infer_out bash ...
#
# Environment:
#   CHECKPOINT_ROOT - where .pt files live (default: <repo>/result_round1)
#   OUTPUT_ROOT     - where csv/imgs are written (default: <repo>/result_offline_infer)
#   PYTHON          - python executable (default: python)
#   SEEDS           - space-separated list (default: 111 222 333 444 555); each needs a checkpoint
#   GPU_ID          - passed to test_cls --gpu-id (default: 0)
#   DATASETS        - space-separated (default: mvtec visa — VisA needs checkpoints under CHECKPOINT_ROOT)
#   SHOTS           - space-separated k-shot values (default: 1 2 4)
#   VIS             - true/false (default: false)
#   SKIP_MISSING    - if 1, skip when checkpoint file is missing (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Back-compat: ROOT_DIR alone = same tree for weights + outputs
if [[ -n "${ROOT_DIR:-}" ]]; then
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT_DIR}}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}}"
else
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/result_round1}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/result_offline_infer}"
fi
PYTHON_BIN="${PYTHON:-python}"
# Multiple seeds: each needs CLS-Seed_${seed}-*.pt (train with those seeds first)
SEEDS="${SEEDS:-111 222 333 444 555}"
GPU_ID="${GPU_ID:-0}"
DATASETS="${DATASETS:-mvtec visa}"
SHOTS="${SHOTS:-1 2 4}"
VIS="${VIS:-false}"
SKIP_MISSING="${SKIP_MISSING:-1}"
TASK="CLS"

export DATASETS
export SHOTS

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "[ERROR] CHECKPOINT_ROOT not found: ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
echo "[INFO] CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] SEEDS=${SEEDS}"

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
skip=0
fail=0

for SEED in ${SEEDS}; do
  while IFS=$'\t' read -r ds cls shot; do
    [[ -z "${ds:-}" ]] && continue
    ckpt="${CHECKPOINT_ROOT}/${ds}/k_${shot}/checkpoint/${TASK}-Seed_${SEED}-${cls}-check_point.pt"
    if [[ ! -f "${ckpt}" ]]; then
      if [[ "${SKIP_MISSING}" == "1" ]]; then
        echo "[SKIP] no checkpoint: ${ckpt}"
        skip=$((skip + 1))
        continue
      fi
      echo "[ERROR] missing checkpoint: ${ckpt}" >&2
      exit 1
    fi

    echo "[RUN] dataset=${ds} class=${cls} k-shot=${shot} seed=${SEED}"
    OUT_ARGS=()
    if [[ "${OUTPUT_ROOT}" != "${CHECKPOINT_ROOT}" ]]; then
      OUT_ARGS+=(--output-root "${OUTPUT_ROOT}")
    fi
    if "${PYTHON_BIN}" test_cls.py \
      --dataset "${ds}" \
      --class_name "${cls}" \  
      --k-shot "${shot}" \
      --root-dir "${CHECKPOINT_ROOT}" \
      "${OUT_ARGS[@]}" \
      --seed "${SEED}" \
      --gpu-id "${GPU_ID}" \
      --vis "${VIS}"; then
      ok=$((ok + 1))
    else
      echo "[FAIL] dataset=${ds} class=${cls} k-shot=${shot} seed=${SEED}" >&2
      fail=$((fail + 1))
    fi
  done <<< "${plan}"
done

echo "[DONE] ok=${ok} skip=${skip} fail=${fail}"
if [[ "${fail}" -gt 0 ]]; then
  exit 2
fi
