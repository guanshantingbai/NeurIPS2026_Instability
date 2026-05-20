#!/usr/bin/env bash
# =============================================================================
# Stage 0 — Preparation (no model scoring)
#
# Sub-steps (extend over time):
#   - Dataset / env path checks
#   - external/ repo presence
#   - PatchCore pretrained weights verify/download
#   - shell + python sanity checks
#
# Reference: docs/PIPELINE_STAGES.md
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

PY="${PYTHON:-python3}"
FAIL=0

echo "=============================================================================="
echo "[stage0] Preparation — no train/infer/export"
echo "=============================================================================="

# --- Shell sanity ---
echo ""
echo "[stage0] bash -n scripts/*.sh"
if ! bash -n "$REPO_ROOT"/scripts/*.sh 2>/dev/null; then
  echo "[stage0] WARN: some scripts/*.sh failed bash -n" >&2
  FAIL=1
fi

# --- Python compile ---
echo "[stage0] python3 -m compileall -q src"
if ! "$PY" -m compileall -q "$REPO_ROOT/src"; then
  echo "[stage0] ERROR: compileall failed" >&2
  FAIL=1
fi

# --- external/ check ---
echo "[stage0] external/ repositories"
for d in PromptAD PaDiM-Anomaly-Detection-Localization-master patchcore-inspection; do
  if [ -d "$REPO_ROOT/external/$d" ]; then
    echo "  OK  external/$d"
  else
    echo "  MISS external/$d" >&2
    FAIL=1
  fi
done

# --- Data path hints (non-fatal unless STRICT_STAGE0=1) ---
echo "[stage0] optional data / model env (informational)"
for var in PATCHCORE_DATA_ROOT PATCHCORE_MODELS_RUN PADIM_DATA_ROOT PROMPTAD_OUTPUT_ROOT; do
  if [ -n "${!var:-}" ]; then
    echo "  set  $var=${!var}"
  else
    echo "  unset $var"
  fi
done

# --- PatchCore weights (placeholder sub-step) ---
echo "[stage0] PatchCore pretrained weights"
if [ -n "${PATCHCORE_MODELS_RUN:-}" ] && [ -d "${PATCHCORE_MODELS_RUN}" ]; then
  echo "  OK  PATCHCORE_MODELS_RUN exists: ${PATCHCORE_MODELS_RUN}"
else
  echo "  NOTE: set PATCHCORE_MODELS_RUN before Stage 1 PatchCore inference"
  echo "        (auto-download hook TBD in this script)"
  if [ "${STRICT_STAGE0:-0}" = "1" ]; then
    FAIL=1
  fi
fi

echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "[stage0] done — ready for Stage 1 (run_*_raw.sh) or Stage 2+3 (rebuild_main_figures.sh with existing raw)"
else
  echo "[stage0] finished with warnings/errors (exit 1)" >&2
  exit 1
fi
