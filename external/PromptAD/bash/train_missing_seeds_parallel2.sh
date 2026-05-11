#!/usr/bin/env bash
# Train + test + pairwise for PromptAD (dataset, class, k, seed) jobs that are missing under
# result_seed_search/{slug}/{seed}/pairwise_instability/summary.json
#
# Concurrency: at most MAX_PARALLEL full pipelines in parallel (default 2).
# Each Python process: BLAS/OpenMP threads capped at 2 (see exports below).
#
# Environment (optional):
#   PROMPTAD          repo root (default: parent of this bash/ directory)
#   SEARCH_ROOT       default ${PROMPTAD}/result_seed_search
#   RESULT_ROUND1     default ${PROMPTAD}/result_round1
#   DESIRED_SEEDS     default "222,333,444,555" (comma or space; add 111 to mirror round1 into seed_search)
#   GPU_ID            default 0
#   EVAL_FREQ         default 2 (train_cls --eval-freq)
#   MAX_PARALLEL      default 2
#   PYTHON            default python3
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPTAD="${PROMPTAD:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SEARCH_ROOT="${SEARCH_ROOT:-${PROMPTAD}/result_seed_search}"
RESULT_ROUND1="${RESULT_ROUND1:-${PROMPTAD}/result_round1}"
DESIRED_SEEDS="${DESIRED_SEEDS:-222,333,444,555}"
GPU_ID="${GPU_ID:-0}"
EVAL_FREQ="${EVAL_FREQ:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
PY="${PYTHON:-python3}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

_emit_jobs() {
  "${PY}" "${PROMPTAD}/utils/emit_missing_promptad_seed_jobs.py" \
    --result-round1 "${RESULT_ROUND1}" \
    --search-root "${SEARCH_ROOT}" \
    --seeds "${DESIRED_SEEDS// /,}"
}

_wait_slot() {
  # Portable: avoid requiring `wait -n` (bash 5+).
  while true; do
    local n
    n="$(jobs -rp 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${n:-0}" -lt "${MAX_PARALLEL}" ]]; then
      break
    fi
    sleep 2
  done
}

_run_one() {
  local dataset="$1" class="$2" k="$3" seed="$4" slug="$5" job_root="$6"
  (
    set -euo pipefail
    cd "${PROMPTAD}"
    echo "=== START ${slug} seed=${seed} ($(date -Is)) ==="
    "${PY}" train_cls.py \
      --dataset "${dataset}" \
      --class_name "${class}" \
      --k-shot "${k}" \
      --root-dir "${job_root}" \
      --seed "${seed}" \
      --gpu-id "${GPU_ID}" \
      --eval-freq "${EVAL_FREQ}" \
      --num-workers 2
    "${PY}" test_cls.py \
      --dataset "${dataset}" \
      --class_name "${class}" \
      --k-shot "${k}" \
      --root-dir "${job_root}" \
      --output-root "${job_root}" \
      --seed "${seed}" \
      --gpu-id "${GPU_ID}" \
      --vis false \
      --num-workers 2
    local csv_path="${job_root}/${dataset}/k_${k}/csv/CLS-${dataset}-${class}-k${k}-seed${seed}-per_sample.csv"
    test -f "${csv_path}"
    "${PY}" utils/pairwise_instability.py \
      --csv_path "${csv_path}" \
      --output_dir "${job_root}/pairwise_instability"
    echo "=== DONE ${slug} seed=${seed} ($(date -Is)) ==="
  ) &
}

main() {
  local jobfile n
  jobfile="$(mktemp)"
  trap '[[ -n "${jobfile:-}" ]] && rm -f "${jobfile}"' EXIT

  _emit_jobs >"${jobfile}" || true
  n="$(wc -l <"${jobfile}" | tr -d ' ')"
  if [[ "${n}" -eq 0 ]]; then
    echo "No missing (dataset, class, k, seed) jobs for seeds [${DESIRED_SEEDS}] under ${SEARCH_ROOT}."
    exit 0
  fi

  echo "Queued ${n} jobs (max ${MAX_PARALLEL} concurrent pipelines). PROMPTAD=${PROMPTAD}"
  while IFS=$'\t' read -r dataset class k seed slug job_root; do
    [[ -n "${dataset:-}" ]] || continue
    _wait_slot
    mkdir -p "${job_root}"
    _run_one "${dataset}" "${class}" "${k}" "${seed}" "${slug}" "${job_root}"
  done <"${jobfile}"

  echo "Waiting for remaining jobs..."
  wait
  echo "All jobs finished."
}

main "$@"
