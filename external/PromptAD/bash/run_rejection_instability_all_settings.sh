#!/usr/bin/env bash
set -euo pipefail

# Batch-run utils/rejection_instability_analysis.py on every exported per-sample CSV
# (one CLS setting per file: dataset × k-shot × class × seed), e.g. 81 tables under
#   result_round1/{mvtec,visa}/k_{1,2,4}/csv/CLS-*-per_sample.csv
#
# Output layout (mirrors pairwise_instability sibling folders):
#   ${ROOT_DIR}/{dataset}/k_{k}/rejection_instability/<CSV_basename_without_.csv>/
#     per_sample_instability_analysis.csv, scatter_*.png, bar_tertile_proxy.png,
#     rejection_*curve*.csv, curve_*.png
#
# Usage (from anywhere):
#   bash /path/to/PromptAD/bash/run_rejection_instability_all_settings.sh
#
# From PromptAD repo root:
#   bash bash/run_rejection_instability_all_settings.sh
#
# Environment:
#   ROOT_DIR            - root containing {mvtec,visa}/k_*/csv/ (default: <repo>/result_round1)
#   PYTHON              - interpreter (default: python)
#   RANDOM_REPEATS      - --random-repeats (default: 20)
#   SEED                - --seed for random rejection baseline (default: 0)
#   CONTINUE_ON_ERROR   - if 1, keep going on failure and exit 2 if any failed (default: 1)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/result_round1}"
PYTHON_BIN="${PYTHON:-python}"
ANALYZER="${ANALYZER:-${REPO_ROOT}/utils/rejection_instability_analysis.py}"
RANDOM_REPEATS="${RANDOM_REPEATS:-20}"
SEED="${SEED:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "[ERROR] ROOT_DIR not found: ${ROOT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ANALYZER}" ]]; then
  echo "[ERROR] Analyzer not found: ${ANALYZER}" >&2
  exit 1
fi

shopt -s nullglob
tables=( "${ROOT_DIR}"/*/k_*/csv/*per_sample.csv )
shopt -u nullglob

if [[ ${#tables[@]} -eq 0 ]]; then
  echo "[ERROR] No per-sample tables under ${ROOT_DIR}/*/k_*/csv/*per_sample.csv" >&2
  exit 1
fi

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] Found ${#tables[@]} per-sample CSV(s)."
echo "[INFO] PYTHON=${PYTHON_BIN} RANDOM_REPEATS=${RANDOM_REPEATS} SEED=${SEED}"

fail_log="${ROOT_DIR}/rejection_instability_failures.txt"
rm -f "${fail_log}" || true

ok=0
fail=0

for csv_path in "${tables[@]}"; do
  rel_dir="$(dirname "$(dirname "${csv_path}")")"
  base="$(basename "${csv_path}" .csv)"
  out_dir="${rel_dir}/rejection_instability/${base}"

  echo "[RUN] (${ok}+${fail}+1/${#tables[@]}) ${csv_path}"
  echo "      -> ${out_dir}"

  set +e
  "${PYTHON_BIN}" "${ANALYZER}" \
    --input-csv "${csv_path}" \
    --output-dir "${out_dir}" \
    --random-repeats "${RANDOM_REPEATS}" \
    --seed "${SEED}"
  rc=$?
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    echo "${csv_path}" >> "${fail_log}"
    echo "[FAIL] rc=${rc} ${csv_path}" >&2
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${rc}"
    fi
  fi
done

echo "[DONE] ok=${ok} fail=${fail} total=${#tables[@]}"
if [[ ${fail} -gt 0 ]]; then
  echo "[WARN] Failure list: ${fail_log}" >&2
  exit 2
fi
