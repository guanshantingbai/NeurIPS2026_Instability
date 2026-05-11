#!/usr/bin/env bash
# Train PromptAD CLS with up to 2 concurrent processes and <=2 CPU threads per process.
#
# Each seed needs its own training run → separate checkpoints (CLS-Seed_${seed}-*.pt).
#
# GPU:
#   - Set GPU_IDS="0 1" to pin jobs to two physical GPUs (recommended with MAX_PARALLEL=2).
#   - Default GPU_IDS="0" runs both workers on GPU 0 (may OOM; reduce MAX_PARALLEL or batch size).
#
# CPU (per training process):
#   OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS / NUMEXPR = 2
#
# Usage:
#   cd PromptAD && bash bash/run_cls_train_parallel2.sh
#
# Env:
#   MAX_PARALLEL   default 2
#   GPU_IDS        default "0" (space-separated, round-robin across jobs)
#   SEEDS          default "111 222 333 444 555"
#   DATASETS       default "mvtec visa"
#   SHOTS          default "1 2 4"
#   ROOT_DIR       default <repo>/result_offline_train
#   INSTABILITY_PENALTY_LAMBDA  optional
#   PYTHON         default python
#
# Requires bash >= 4.3 (wait -n). Logs are kept under a temp dir printed at the end.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MAX_PARALLEL="${MAX_PARALLEL:-2}"
read -r -a GPU_IDS_ARR <<< "${GPU_IDS:-0}"

ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/result_offline_train}"
PYTHON_BIN="${PYTHON:-python}"
SEEDS="${SEEDS:-111 222 333 444 555}"
DATASETS="${DATASETS:-mvtec visa}"
SHOTS="${SHOTS:-1 2 4}"
EVAL_FREQ="${EVAL_FREQ:-2}"

export DATASETS
export SEEDS
export SHOTS

mkdir -p "${ROOT_DIR}"

EXTRA_ARGS=()
if [[ -n "${INSTABILITY_PENALTY_LAMBDA:-}" ]]; then
  EXTRA_ARGS+=(--instability-penalty-lambda "${INSTABILITY_PENALTY_LAMBDA}")
fi

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] MAX_PARALLEL=${MAX_PARALLEL}  GPU_IDS=${GPU_IDS_ARR[*]}"
echo "[INFO] SEEDS=${SEEDS}"
echo "[INFO] Per-job CPU threads: OMP/MKL/OPENBLAS/NUMEXPR=2"

TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/promptad_train_XXXXXX")"
trap 'echo "[WARN] interrupted; logs in ${TMPDIR}" >&2' INT TERM

plan="$(
  "${PYTHON_BIN}" - <<'PY'
import os
from datasets import dataset_classes

datasets = os.environ["DATASETS"].split()
shots = os.environ["SHOTS"].split()
seeds = os.environ["SEEDS"].split()
for seed in seeds:
    for ds in datasets:
        if ds not in dataset_classes:
            raise SystemExit(f"unknown dataset: {ds!r}")
        for shot in shots:
            for cls in dataset_classes[ds]:
                print(f"{ds}\t{cls}\t{shot}\t{seed}")
PY
)"

job_idx=0
while IFS=$'\t' read -r ds cls shot seed; do
  [[ -z "${ds:-}" ]] && continue

  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n 2>/dev/null || wait
  done

  phys="${GPU_IDS_ARR[$((job_idx % ${#GPU_IDS_ARR[@]}))]}"
  key="${ds}__${cls}__k${shot}__s${seed}"
  key="${key//[^a-zA-Z0-9_]/_}"
  logf="${TMPDIR}/train_${key}.log"

  echo "[QUEUE] -> GPU ${phys}  ${ds} ${cls} k=${shot} seed=${seed}  (log: ${logf})"

  (
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    export NUMEXPR_NUM_THREADS=2
    export CUDA_VISIBLE_DEVICES="${phys}"
    cd "${REPO_ROOT}"
    if "${PYTHON_BIN}" train_cls.py \
      --dataset "${ds}" \
      --class_name "${cls}" \
      --k-shot "${shot}" \
      --root-dir "${ROOT_DIR}" \
      --seed "${seed}" \
      --gpu-id 0 \
      --eval-freq "${EVAL_FREQ}" \
      "${EXTRA_ARGS[@]}" >"${logf}" 2>&1; then
      echo "OK	${ds}	${cls}	${shot}	${seed}" >"${TMPDIR}/rc_${job_idx}.txt"
    else
      echo "FAIL	${ds}	${cls}	${shot}	${seed}" >"${TMPDIR}/rc_${job_idx}.txt"
    fi
  ) &

  job_idx=$((job_idx + 1))
done <<< "${plan}"

wait || true

ok=0
fail=0
for f in "${TMPDIR}"/rc_*.txt; do
  [[ -f "$f" ]] || continue
  if head -1 "$f" | grep -q '^OK'; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    echo "[FAIL] $(cat "$f")" >&2
  fi
done

echo "[DONE] ok=${ok} fail=${fail}"
echo "[INFO] Per-job logs: ${TMPDIR}"
if [[ "${fail}" -gt 0 ]]; then
  exit 2
fi
