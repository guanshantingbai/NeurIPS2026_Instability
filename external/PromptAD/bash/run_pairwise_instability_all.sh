#!/usr/bin/env bash
set -euo pipefail

# Traverse all per-sample tables under:
#   ./result_round1/{dataset}/{k}/csv/*per_sample.csv
# and run pairwise instability analysis for each table.
#
# Outputs are saved under:
#   ./result_round1/{dataset}/{k}/pairwise_instability/<table_basename>/
#
# Usage:
#   bash ./bash/run_pairwise_instability_all.sh
#   ROOT_DIR=./result_round1 PYTHON=python bash ./bash/run_pairwise_instability_all.sh

ROOT_DIR="${ROOT_DIR:-./result_round1}"
PYTHON_BIN="${PYTHON:-python}"
ANALYZER="${ANALYZER:-./utils/pairwise_instability.py}"

if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "[ERROR] ROOT_DIR not found: ${ROOT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ANALYZER}" ]]; then
  echo "[ERROR] Analyzer script not found: ${ANALYZER}" >&2
  exit 1
fi

shopt -s nullglob
tables=( "${ROOT_DIR}"/*/k_*/csv/*per_sample.csv )
shopt -u nullglob

if [[ ${#tables[@]} -eq 0 ]]; then
  echo "[ERROR] No per-sample tables found under ${ROOT_DIR}/*/k_*/csv/*per_sample.csv" >&2
  exit 1
fi

echo "[INFO] Found ${#tables[@]} tables."

fail_log="${ROOT_DIR}/pairwise_instability_failures.txt"
rm -f "${fail_log}" || true

ok=0
fail=0

for csv_path in "${tables[@]}"; do
  # Example:
  #   result_round1/visa/k_2/csv/CLS-visa-capsules-k2-seed111-per_sample.csv
  # Derive output dir:
  #   result_round1/visa/k_2/pairwise_instability/CLS-visa-capsules-k2-seed111-per_sample/
  rel_dir="$(dirname "$(dirname "${csv_path}")")"  # .../{dataset}/{k}
  base="$(basename "${csv_path}" .csv)"
  out_dir="${rel_dir}/pairwise_instability/${base}"

  echo "[RUN] ${csv_path}"
  echo "      -> ${out_dir}"

  if "${PYTHON_BIN}" "${ANALYZER}" --csv_path "${csv_path}" --output_dir "${out_dir}"; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    echo "${csv_path}" >> "${fail_log}"
    echo "[FAIL] ${csv_path}" >&2
  fi
done

echo "[DONE] ok=${ok} fail=${fail}"
if [[ ${fail} -gt 0 ]]; then
  echo "[WARN] Failure list saved to: ${fail_log}" >&2
  exit 2
fi

