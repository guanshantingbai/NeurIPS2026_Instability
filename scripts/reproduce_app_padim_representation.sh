#!/usr/bin/env bash
# Stage 2: Appendix E PaDiM representation (consumes raw-derived CSV or fast-path stub only).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"
bash src/experiments/app_padim_representation/run.sh
