#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

echo "[main] section 3.1.1 promptad observation (fast path unless FULL_RUN=1)"
bash scripts/reproduce_sec3_promptad.sh

echo "[main] section 3.1.2 padim observation (fast path unless FULL_RUN=1)"
bash scripts/reproduce_sec3_padim.sh

echo "[main] section 4 systematic validation (local stubs; FULL_RUN=1 for external strengthening)"
bash scripts/reproduce_sec4_systematic.sh

echo "[main] done"
