#!/usr/bin/env bash
# Fetch real PatchCore MVTec checkpoints (replace Git LFS pointer files).
# Requires: git + git-lfs on PATH, e.g.
#   conda install -n myenv -c conda-forge git-lfs
#   export PATH="/path/to/miniconda3/envs/myenv/bin:$PATH"
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git lfs install
git lfs pull
echo "Done. Spot-check (pointer files are ~130 bytes; real weights are MB+):"
wc -c models/IM320_WR50_L2-3_P001_D1024-1024_PS-3_AN-1/models/mvtec_bottle/patchcore_params.pkl
