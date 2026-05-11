#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

echo "[appendix] C promptad generalization (fast path unless FULL_RUN=1)"
bash src/experiments/app_promptad_generalization/run.sh

echo "[appendix] E padim representation (Stage 2 only; see docs/REPRODUCE.md)"
bash scripts/reproduce_app_padim_representation.sh

echo "[appendix] F patchcore tta (Stage 2 only; see docs/REPRODUCE.md)"
bash scripts/reproduce_app_patchcore_tta.sh

echo "[appendix] G signal comparison (bundled stub unless FULL_RUN=1)"
bash src/experiments/app_signal_comparison/run.sh

echo "[appendix] done"
