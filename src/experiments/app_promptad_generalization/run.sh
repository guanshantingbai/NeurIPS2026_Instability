#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

mkdir -p outputs/figures/app_promptad_generalization outputs/tables/app_promptad_generalization outputs/cached_results/app_promptad_generalization

if [ "${FULL_RUN:-0}" = "1" ]; then
  echo "WARNING: FULL_RUN=1 runs PromptAD seed_killer pipeline (phase1/2/3 as configured; heavy)." >&2
  echo "         Pass extra args after setting FULL_RUN, e.g. phase3 --proxy u6" >&2
  python -m src.models.promptad_adapter.run_promptad "utils/seed_killer_evidence_pipeline.py" "$@"
  exit 0
fi

SAMPLE="$REPO_ROOT/src/experiments/app_promptad_generalization/samples/fastpath"
if [ ! -d "$SAMPLE/tables" ]; then
  echo "ERROR: Appendix C fast path missing $SAMPLE/tables" >&2
  echo "       Set FULL_RUN=1 to run external seed_killer (requires prior artifacts)." >&2
  exit 1
fi

shopt -s nullglob
tbl=("$SAMPLE/tables/"*.csv)
if [ "${#tbl[@]}" -eq 0 ]; then
  echo "ERROR: no CSV under $SAMPLE/tables" >&2
  exit 1
fi
cp -f "${tbl[@]}" outputs/tables/app_promptad_generalization/
echo "fastpath_done" >outputs/figures/app_promptad_generalization/fastpath_done.txt
echo "app_promptad_generalization: fast path (cached-only samples) OK"
