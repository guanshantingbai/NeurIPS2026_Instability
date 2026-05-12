#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

mkdir -p outputs/figures/sec4_systematic outputs/tables/sec4_systematic outputs/cached_results/sec4_systematic

RAW_DIR="$REPO_ROOT/outputs/cached_results/raw_scores/promptad"
WIDE="$RAW_DIR/unified_raw_scores_wide.csv"
LONG="$RAW_DIR/unified_raw_scores_long.csv"
if [ -f "$WIDE" ] || [ -f "$LONG" ]; then
  echo "sec4_systematic: building from PromptAD unified raw under $RAW_DIR (pairwise; no external PromptAD scripts)"
  PY="${PYTHON:-python3}"
  SEC4_EXTRA=()
  if [ -n "${SEC4_MAX_PAIRS_PER_SETTING:-}" ]; then
    SEC4_EXTRA+=(--max-pairs-per-setting "${SEC4_MAX_PAIRS_PER_SETTING}")
  fi
  if [ -n "${SEC4_PAIR_SAMPLING_SEED:-}" ]; then
    SEC4_EXTRA+=(--pair-sampling-seed "${SEC4_PAIR_SAMPLING_SEED}")
  fi
  "$PY" "$SCRIPT_DIR/analyze_sec4_promptad_from_raw.py" --raw-dir "$RAW_DIR" "${SEC4_EXTRA[@]}"
  echo "sec4_systematic: from PromptAD raw OK"
  exit 0
fi

python "$SCRIPT_DIR/build_candidate_pairs.py"
python "$SCRIPT_DIR/compute_margin_analysis.py"
python "$SCRIPT_DIR/compute_failure_signals.py"
python "$SCRIPT_DIR/compute_delta_risk.py"

if [ "${FULL_RUN:-0}" = "1" ]; then
  echo "WARNING: FULL_RUN=1 runs PromptAD strengthening experiments (heavy; needs pilot CSVs under external/PromptAD)." >&2
  python -m src.models.promptad_adapter.run_promptad "utils/promptad_strengthening_experiments.py"
fi

python "$SCRIPT_DIR/plot_sec4_figures.py"
echo "sec4_systematic: fast path (local stubs) OK; set FULL_RUN=1 for external strengthening pipeline"
