#!/usr/bin/env bash
# Stage 2 only: never runs scripts/run_padim_raw.sh (Stage 1).
# Consumes mechanism_from_raw.csv if present, else bundled fast-path stub.
# PADIM_FROM_RAW=1 requires mechanism_from_raw.csv or errors with Stage 1 hint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

mkdir -p outputs/figures/app_padim_representation outputs/tables/app_padim_representation outputs/cached_results/app_padim_representation

MECH="$REPO_ROOT/outputs/cached_results/app_padim_representation/mechanism_from_raw.csv"
SAMPLE="$REPO_ROOT/src/experiments/app_padim_representation/samples/fastpath"

if [ -f "$MECH" ]; then
  cp -f "$MECH" outputs/tables/app_padim_representation/
  echo "stage2_padim_appendix_e" >outputs/figures/app_padim_representation/fullrun_done.txt
  echo "app_padim_representation: Stage 2 OK (mechanism_from_raw.csv -> tables)"
  exit 0
fi

if [ "${PADIM_FROM_RAW:-0}" = "1" ]; then
  echo "ERROR: PADIM_FROM_RAW=1 but missing $MECH" >&2
  echo "       Run Stage 1 first: FULL_RUN=1 bash scripts/run_padim_raw.sh" >&2
  echo "       (see docs/FULLPATH_PADIM.md)" >&2
  exit 1
fi

if [ ! -d "$SAMPLE/tables" ]; then
  echo "ERROR: Appendix E fast path missing $SAMPLE/tables" >&2
  echo "       Provide mechanism_from_raw.csv after Stage 1, or restore bundled stubs." >&2
  exit 1
fi

shopt -s nullglob
tbl=("$SAMPLE/tables/"*.csv)
if [ "${#tbl[@]}" -eq 0 ]; then
  echo "ERROR: no CSV under $SAMPLE/tables" >&2
  exit 1
fi
cp -f "${tbl[@]}" outputs/tables/app_padim_representation/
echo "fastpath_done" >outputs/figures/app_padim_representation/fastpath_done.txt
echo "app_padim_representation: fast path (cached-only samples) OK"
