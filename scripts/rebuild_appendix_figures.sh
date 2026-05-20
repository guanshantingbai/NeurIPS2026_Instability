#!/usr/bin/env bash
# =============================================================================
# Stage 4 — Appendix figure generation (PARTIAL)
#
# Target: outputs/figures/app_{patchcore_tta,padim_representation,promptad_generalization,signal_comparison}/...
#
# Current behavior: delegates to scripts/reproduce_appendix.sh (legacy Stage-2-style
# per-appendix fast path / cached raw). A dedicated Stage-4-only figure bundle is TBD.
#
# Does NOT run Stage 0, 1, or main-paper Stage 2+3 (use rebuild_main_figures.sh).
#
# Reference: docs/PIPELINE_STAGES.md
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

echo "=============================================================================="
echo "[rebuild_appendix] Stage 4 — PARTIAL"
echo "  Planned outputs: outputs/figures/app_*"
echo "  Current: reproduce_appendix.sh (fast path / cached assets; no Stage 1 auto-run)"
echo "=============================================================================="

bash "$REPO_ROOT/scripts/reproduce_appendix.sh"

echo ""
echo "[rebuild_appendix] done (partial Stage 4 via reproduce_appendix.sh)."
echo "  See docs/FIGURE_MAP.md for per-appendix paths."
