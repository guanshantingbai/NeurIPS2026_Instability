#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

SEC3_DONE_MARKER="$REPO_ROOT/outputs/figures/sec3_promptad/from_raw_done.txt"
SEC3_PAIRWISE="$REPO_ROOT/outputs/cached_results/sec3_promptad/pairwise_metrics.csv"
SEC4_DONE_MARKER="$REPO_ROOT/outputs/figures/sec4_systematic/from_promptad_raw_done.txt"
SEC4_SUMMARY="$REPO_ROOT/outputs/cached_results/sec4_systematic/sec4_systematic_from_raw_summary.json"

# Set FORCE_REBUILD=1 to always run Sec3 / Sec4 reproduce scripts (full recompute paths).
skip_sec3_promptad_from_raw() {
  [ "${FORCE_REBUILD:-0}" = "1" ] && return 1
  [ -f "$SEC3_DONE_MARKER" ] && [ -f "$SEC3_PAIRWISE" ]
}

skip_sec4_systematic_from_raw() {
  [ "${FORCE_REBUILD:-0}" = "1" ] && return 1
  [ -f "$SEC4_DONE_MARKER" ] && [ -f "$SEC4_SUMMARY" ]
}

echo "[main] section 3.1.1 promptad observation (fast path unless FULL_RUN=1)"
if skip_sec3_promptad_from_raw; then
  echo "[main] SKIP reproduce_sec3_promptad.sh (cache hit): $SEC3_DONE_MARKER and $SEC3_PAIRWISE exist (set FORCE_REBUILD=1 to force full Sec3 recompute)"
else
  bash scripts/reproduce_sec3_promptad.sh
fi

echo "[main] section 3.1.2 padim observation (fast path unless FULL_RUN=1)"
bash scripts/reproduce_sec3_padim.sh

echo "[main] section 4 systematic validation (local stubs; FULL_RUN=1 for external strengthening)"
if skip_sec4_systematic_from_raw; then
  echo "[main] SKIP reproduce_sec4_systematic.sh (cache hit): $SEC4_DONE_MARKER and $SEC4_SUMMARY exist (set FORCE_REBUILD=1 to force full Sec4 recompute)"
else
  bash scripts/reproduce_sec4_systematic.sh
fi

echo "[main] done"
