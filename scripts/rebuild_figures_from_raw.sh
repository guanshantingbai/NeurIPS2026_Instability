#!/usr/bin/env bash
# DEPRECATED: use scripts/rebuild_main_figures.sh (see docs/PIPELINE_STAGES.md).
set -euo pipefail
echo "[DEPRECATED] scripts/rebuild_figures_from_raw.sh was renamed to scripts/rebuild_main_figures.sh" >&2
echo "  This wrapper will be removed in a future cleanup. Please update your commands." >&2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rebuild_main_figures.sh" "$@"
