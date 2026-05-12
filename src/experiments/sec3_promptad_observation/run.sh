#!/usr/bin/env bash
# Section 3.1.1: prefer unified PromptAD raw scores (Stage 1 export). No train/infer.
# Optional stub: SEC3_PROMPTAD_ALLOW_STUB=1 (CI / demos without raw).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

mkdir -p outputs/figures/sec3_promptad outputs/cached_results/sec3_promptad

PY="${PYTHON:-python3}"
RAW_DIR="$REPO_ROOT/outputs/cached_results/raw_scores/promptad"
WIDE="$RAW_DIR/unified_raw_scores_wide.csv"
LONG="$RAW_DIR/unified_raw_scores_long.csv"

if [ -f "$WIDE" ] || [ -f "$LONG" ]; then
  echo "sec3_promptad: building from unified raw scores under $RAW_DIR"
  SEC3_EXTRA=()
  if [ -n "${SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING:-}" ]; then
    SEC3_EXTRA+=(--max-pairs-per-setting "${SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING}")
  fi
  if [ -n "${SEC3_PROMPTAD_PAIR_SAMPLING_SEED:-}" ]; then
    SEC3_EXTRA+=(--pair-sampling-seed "${SEC3_PROMPTAD_PAIR_SAMPLING_SEED}")
  fi
  "$PY" "$SCRIPT_DIR/analyze_sec3_promptad_from_raw.py" --raw-dir "$RAW_DIR" "${SEC3_EXTRA[@]}"
  echo "sec3_promptad: from raw OK"
  exit 0
fi

if [ "${SEC3_PROMPTAD_ALLOW_STUB:-0}" = "1" ]; then
  SAMPLE="$REPO_ROOT/src/experiments/sec3_promptad_observation/samples/fastpath"
  if [ ! -d "$SAMPLE/cached_results" ]; then
    echo "ERROR: SEC3_PROMPTAD_ALLOW_STUB=1 but missing $SAMPLE/cached_results" >&2
    exit 1
  fi
  shopt -s nullglob
  stub_csv=("$SAMPLE/cached_results/"*.csv)
  if [ "${#stub_csv[@]}" -eq 0 ]; then
    echo "ERROR: no CSV under $SAMPLE/cached_results" >&2
    exit 1
  fi
  cp -f "${stub_csv[@]}" outputs/cached_results/sec3_promptad/
  "$PY" "$SCRIPT_DIR/plot_promptad_observation.py"
  echo "sec3_promptad: stub fallback (SEC3_PROMPTAD_ALLOW_STUB=1) OK"
  exit 0
fi

echo "ERROR: missing PromptAD unified raw scores:" >&2
echo "       $WIDE (or $LONG)" >&2
echo "       Run Stage 1 export first: FULL_RUN=1 PROMPTAD_MODE=export PROMPTAD_OUTPUT_ROOT=... bash scripts/run_promptad_raw.sh" >&2
echo "       Or set SEC3_PROMPTAD_ALLOW_STUB=1 to copy bundled pairwise_stub.csv (non-empirical)." >&2
exit 1
