#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

mkdir -p outputs/figures/app_padim_representation outputs/tables/app_padim_representation outputs/cached_results/app_padim_representation

if [ "${FULL_RUN:-0}" = "1" ]; then
  echo "WARNING: FULL_RUN=1 runs PaDiM seed_killer_evidence_pipeline (heavy)." >&2
  python -m src.models.padim_adapter.run_padim "padim_seed_killer_evidence_pipeline.py" "$@"
  exit 0
fi

SAMPLE="$REPO_ROOT/src/experiments/app_padim_representation/samples/fastpath"
if [ ! -d "$SAMPLE/tables" ]; then
  echo "ERROR: Appendix E fast path missing $SAMPLE/tables" >&2
  echo "       Set FULL_RUN=1 for full PaDiM appendix pipeline." >&2
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
