#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

mkdir -p outputs/figures/app_signal_comparison outputs/tables/app_signal_comparison outputs/cached_results/app_signal_comparison

STUB_ROOT="$REPO_ROOT/src/experiments/app_signal_comparison/samples/fastpath/promptad_stub"
PILOT_DIR="$REPO_ROOT/src/experiments/app_signal_comparison/samples/fastpath/pilot_instability_selection"

if [ "${FULL_RUN:-0}" = "1" ]; then
  echo "WARNING: FULL_RUN=1 runs supplementary_signal_baselines on your real pilot paths (set PROMPTAD_ROOT / PILOT_DIR if needed)." >&2
  PROMPTAD_R="${PROMPTAD_ROOT:-$REPO_ROOT/external/PromptAD}"
  PILOT="${PILOT_DIR_OVERRIDE:-$REPO_ROOT/external/PromptAD/result_analysis/pilot_instability_selection}"
  python -m src.models.promptad_adapter.run_promptad "utils/supplementary_signal_baselines.py" \
    --promptad-root "$PROMPTAD_R" \
    --pilot-dir "$PILOT" \
    --out-dir "$REPO_ROOT/outputs/figures/app_signal_comparison"
else
  if [ ! -d "$STUB_ROOT/result_seed_search" ] || [ ! -f "$PILOT_DIR/failure_driven/failure_analysis.csv" ]; then
    echo "ERROR: Appendix G fast path samples missing under $STUB_ROOT or $PILOT_DIR" >&2
    exit 1
  fi
  python -m src.models.promptad_adapter.run_promptad "utils/supplementary_signal_baselines.py" \
    --promptad-root "$STUB_ROOT" \
    --pilot-dir "$PILOT_DIR" \
    --out-dir "$REPO_ROOT/outputs/figures/app_signal_comparison"
fi

shopt -s nullglob
csvf=(outputs/figures/app_signal_comparison/*.csv)
if [ "${#csvf[@]}" -gt 0 ]; then
  cp -f "${csvf[@]}" outputs/tables/app_signal_comparison/
fi

echo "app_signal_comparison: OK"
