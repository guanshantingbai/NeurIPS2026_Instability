#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

mkdir -p outputs/figures/sec3_promptad outputs/cached_results/sec3_promptad

if [ "${FULL_RUN:-0}" = "1" ]; then
  echo "WARNING: FULL_RUN=1 runs PromptAD seed_killer phase3 (requires existing result_seed_search artifacts)." >&2
  python -m src.models.promptad_adapter.run_promptad "utils/seed_killer_evidence_pipeline.py" phase3 --proxy u6
  python "$SCRIPT_DIR/plot_promptad_observation.py"
  exit 0
fi

SAMPLE="$REPO_ROOT/src/experiments/sec3_promptad_observation/samples/fastpath"
if [ ! -d "$SAMPLE/cached_results" ]; then
  echo "ERROR: sec3_promptad fast path missing samples under $SAMPLE/cached_results" >&2
  echo "       Install samples or set FULL_RUN=1 (requires prior PromptAD outputs; see docs/REPRODUCIBILITY_STATUS.md)." >&2
  exit 1
fi

shopt -s nullglob
stub_csv=("$SAMPLE/cached_results/"*.csv)
if [ "${#stub_csv[@]}" -eq 0 ]; then
  echo "ERROR: no CSV files under $SAMPLE/cached_results" >&2
  exit 1
fi
cp -f "${stub_csv[@]}" outputs/cached_results/sec3_promptad/
python "$SCRIPT_DIR/plot_promptad_observation.py"
echo "sec3_promptad: fast path (cached-only samples) OK"
