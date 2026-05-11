#!/usr/bin/env bash
# Stage 2 only: never runs scripts/run_padim_raw.sh (Stage 1).
# Priority: marginal_protocol_b.csv -> plot; else PADIM_FROM_RAW=1 checks raw evidence;
# else fast path stub from samples/fastpath.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

mkdir -p outputs/figures/sec3_padim outputs/cached_results/sec3_padim

PY="${PYTHON:-python3}"
MARGINAL="$REPO_ROOT/outputs/cached_results/sec3_padim/marginal_protocol_b.csv"
RAW_LONG="$REPO_ROOT/outputs/cached_results/raw_scores/padim/unified_raw_scores_long.csv"
SAMPLE="$REPO_ROOT/src/experiments/sec3_padim_observation/samples/fastpath"

if [ -f "$MARGINAL" ]; then
  echo "sec3_padim: Stage 2 using cached marginal_protocol_b.csv (plot only)."
  "$PY" "$SCRIPT_DIR/plot_padim_observation.py"
  echo "sec3_padim: Stage 2 OK"
  exit 0
fi

if [ "${PADIM_FROM_RAW:-0}" = "1" ]; then
  if [ ! -f "$RAW_LONG" ]; then
    echo "ERROR: PADIM_FROM_RAW=1 but raw evidence missing: $RAW_LONG" >&2
    echo "       Run Stage 1 first: FULL_RUN=1 bash scripts/run_padim_raw.sh" >&2
    echo "       (PADIM_DATA_ROOT, PADIM_OUTPUT_ROOT, PADIM_CLASSES, PADIM_BACKBONES, PADIM_SEEDS; see docs/FULLPATH_PADIM.md)" >&2
    exit 1
  fi
  echo "ERROR: PADIM_FROM_RAW=1 and unified raw exists, but marginal_protocol_b.csv is missing under outputs/cached_results/sec3_padim/." >&2
  echo "       Stage 1 must complete through aggregation (end of run_padim_raw.sh) so marginal_protocol_b.csv is produced." >&2
  echo "       Re-run: FULL_RUN=1 bash scripts/run_padim_raw.sh" >&2
  exit 1
fi

if [ ! -d "$SAMPLE/cached_results" ]; then
  echo "ERROR: sec3_padim fast path missing samples under $SAMPLE/cached_results" >&2
  echo "       Stage 2 needs either marginal_protocol_b.csv (after Stage 1) or bundled stubs." >&2
  exit 1
fi

shopt -s nullglob
stub_csv=("$SAMPLE/cached_results/"*.csv)
if [ "${#stub_csv[@]}" -eq 0 ]; then
  echo "ERROR: no CSV under $SAMPLE/cached_results" >&2
  exit 1
fi
cp -f "${stub_csv[@]}" outputs/cached_results/sec3_padim/
"$PY" "$SCRIPT_DIR/plot_padim_observation.py"
echo "sec3_padim: fast path (cached-only samples) OK"
