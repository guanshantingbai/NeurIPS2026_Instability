#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

mkdir -p outputs/figures/sec3_padim outputs/cached_results/sec3_padim

if [ "${FULL_RUN:-0}" = "1" ]; then
  echo "WARNING: FULL_RUN=1 runs PaDiM one-click seed-killer (long GPU / multi-seed)." >&2
  python -m src.models.padim_adapter.run_padim "run_padim_seed_killer_one_click.sh"
  python "$SCRIPT_DIR/plot_padim_observation.py"
  exit 0
fi

SAMPLE="$REPO_ROOT/src/experiments/sec3_padim_observation/samples/fastpath"
if [ ! -d "$SAMPLE/cached_results" ]; then
  echo "ERROR: sec3_padim fast path missing samples under $SAMPLE/cached_results" >&2
  echo "       Set FULL_RUN=1 for full PaDiM Protocol B (not default; see docs/REPRODUCIBILITY_STATUS.md)." >&2
  exit 1
fi

shopt -s nullglob
stub_csv=("$SAMPLE/cached_results/"*.csv)
if [ "${#stub_csv[@]}" -eq 0 ]; then
  echo "ERROR: no CSV under $SAMPLE/cached_results" >&2
  exit 1
fi
cp -f "${stub_csv[@]}" outputs/cached_results/sec3_padim/
python "$SCRIPT_DIR/plot_padim_observation.py"
echo "sec3_padim: fast path (cached-only samples) OK"
