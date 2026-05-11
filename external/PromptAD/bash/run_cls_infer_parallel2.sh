#!/usr/bin/env bash
# Run test_cls.py with up to 2 concurrent jobs on the same GPU (default: GPU 0).
#
# Per process: OMP/MKL/OPENBLAS/NUMEXPR = 2 (CPU thread cap). DataLoader workers:
#   NUM_WORKERS (default 2).
#
# Usage (from repo root):
#   bash bash/run_cls_infer_parallel2.sh
#
# Full grid by default: (mvtec + visa) × (k-shot 1, 2, 4) × SEEDS.
# If you pass DATASETS="mvtec" or SHOTS="1" you only get that subset (see your shell env).
#
# Env (common overrides):
#   CHECKPOINT_ROOT   default <repo>/result_round1
#   OUTPUT_ROOT       default <repo>/result_offline_infer  (set same as CHECKPOINT_ROOT to write in-place)
#   GPU_IDS           default "0" — space-separated, round-robin; single "0" => both workers share GPU 0
#   MAX_PARALLEL      default 2
#   SEEDS             default "111" — space-separated; each needs CLS-Seed_<seed>-*.pt
#   DATASETS          default "mvtec visa"  (omit override for both datasets)
#   SHOTS             default "1 2 4"       (omit override for all k-shot)
#   VIS               default false
#   SKIP_MISSING      default 1
#   NUM_WORKERS       default 2  (test_cls --num-workers)
#   INSTABILITY_PENALTY_LAMBDA      default 0.1
#   INSTABILITY_THRESHOLD_QUANTILE  default 0.8
#   INSTABILITY_CORRECTION          default h2  (h2 | thresholded); ignored if FUSION_ALPHA>0
#   INSTABILITY_FUSION_ALPHA        default 0  (Strategy A; e.g. 1.0 enables weighted harmonic fusion)
#   PYTHON            default python
#
# Requires bash >= 4.3 (wait -n).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${ROOT_DIR:-}" ]]; then
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ROOT_DIR}}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}}"
else
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/result_round1}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/result_offline_infer}"
fi

MAX_PARALLEL="${MAX_PARALLEL:-2}"
read -r -a GPU_IDS_ARR <<< "${GPU_IDS:-0}"
PYTHON_BIN="${PYTHON:-python}"
SEEDS="${SEEDS:-111}"
DATASETS="${DATASETS:-mvtec visa}"
SHOTS="${SHOTS:-1 2 4}"
VIS="${VIS:-false}"
SKIP_MISSING="${SKIP_MISSING:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TASK="CLS"

INSTABILITY_PENALTY_LAMBDA="${INSTABILITY_PENALTY_LAMBDA:-0.1}"
INSTABILITY_THRESHOLD_QUANTILE="${INSTABILITY_THRESHOLD_QUANTILE:-0.8}"
INSTABILITY_CORRECTION="${INSTABILITY_CORRECTION:-h2}"
INSTABILITY_FUSION_ALPHA="${INSTABILITY_FUSION_ALPHA:-0}"

export DATASETS
export SHOTS
export SEEDS

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "[ERROR] CHECKPOINT_ROOT not found: ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[INFO] CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] MAX_PARALLEL=${MAX_PARALLEL}  GPU_IDS=${GPU_IDS_ARR[*]}"
echo "[INFO] SEEDS=${SEEDS}"
echo "[INFO] Per-job CPU threads: OMP/MKL/OPENBLAS/NUMEXPR=2  num_workers=${NUM_WORKERS}"

TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/promptad_infer_XXXXXX")"
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
            raise SystemExit(f"unknown dataset in DATASETS: {ds!r}")
        for shot in shots:
            for cls in dataset_classes[ds]:
                print(f"{ds}\t{cls}\t{shot}\t{seed}")
PY
)"

job_idx=0
ok=0
skip=0
fail=0

while IFS=$'\t' read -r ds cls shot seed; do
  [[ -z "${ds:-}" ]] && continue

  ckpt="${CHECKPOINT_ROOT}/${ds}/k_${shot}/checkpoint/${TASK}-Seed_${seed}-${cls}-check_point.pt"
  if [[ ! -f "${ckpt}" ]]; then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "[SKIP] no checkpoint: ${ckpt}"
      skip=$((skip + 1))
      continue
    fi
    echo "[ERROR] missing checkpoint: ${ckpt}" >&2
    exit 1
  fi

  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n 2>/dev/null || wait
  done

  phys="${GPU_IDS_ARR[$((job_idx % ${#GPU_IDS_ARR[@]}))]}"
  key="${ds}__${cls}__k${shot}__s${seed}"
  key="${key//[^a-zA-Z0-9_]/_}"
  logf="${TMPDIR}/infer_${key}.log"

  echo "[QUEUE] -> GPU ${phys}  ${ds} ${cls} k=${shot} seed=${seed}  (log: ${logf})"

  OUT_ARGS=()
  if [[ "${OUTPUT_ROOT}" != "${CHECKPOINT_ROOT}" ]]; then
    OUT_ARGS+=(--output-root "${OUTPUT_ROOT}")
  fi

  (
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    export NUMEXPR_NUM_THREADS=2
    export CUDA_VISIBLE_DEVICES="${phys}"
    cd "${REPO_ROOT}"
    if "${PYTHON_BIN}" test_cls.py \
      --dataset "${ds}" \
      --class_name "${cls}" \
      --k-shot "${shot}" \
      --root-dir "${CHECKPOINT_ROOT}" \
      "${OUT_ARGS[@]}" \
      --seed "${seed}" \
      --gpu-id 0 \
      --vis "${VIS}" \
      --num-workers "${NUM_WORKERS}" \
      --instability-penalty-lambda "${INSTABILITY_PENALTY_LAMBDA}" \
      --instability-threshold-quantile "${INSTABILITY_THRESHOLD_QUANTILE}" \
      --instability-correction "${INSTABILITY_CORRECTION}" \
      --instability-fusion-alpha "${INSTABILITY_FUSION_ALPHA}" \
      >"${logf}" 2>&1; then
      echo "OK	${ds}	${cls}	${shot}	${seed}" >"${TMPDIR}/rc_${job_idx}.txt"
    else
      echo "FAIL	${ds}	${cls}	${shot}	${seed}" >"${TMPDIR}/rc_${job_idx}.txt"
    fi
  ) &

  job_idx=$((job_idx + 1))
done <<< "${plan}"

wait || true

for f in "${TMPDIR}"/rc_*.txt; do
  [[ -f "$f" ]] || continue
  if head -1 "$f" | grep -q '^OK'; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    echo "[FAIL] $(cat "$f")" >&2
  fi
done

echo "[DONE] ok=${ok} skip=${skip} fail=${fail}"
echo "[INFO] Per-job logs: ${TMPDIR}"
if [[ "${fail}" -gt 0 ]]; then
  exit 2
fi
