#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

echo "[reproduce_sec3_padim] PADIM_PROFILE=${PADIM_PROFILE:-'(unset)'}  (Stage 2 only; export PADIM_PROFILE in the same shell as Stage 1 if you want it echoed here)"

MARG="${REPO_ROOT}/outputs/cached_results/sec3_padim/marginal_protocol_b.csv"
STUB="${REPO_ROOT}/outputs/cached_results/sec3_padim/marginal_stub.csv"
if [ -f "$MARG" ]; then
  "${PYTHON:-python3}" <<'PY'
import os
from pathlib import Path
import pandas as pd

p = Path(os.environ["REPO_ROOT"]) / "outputs/cached_results/sec3_padim/marginal_protocol_b.csv"
df = pd.read_csv(p)
cats = set()
seeds = set()
for s in df["setting"].astype(str):
    parts = s.split("__")
    if len(parts) == 3:
        cats.add(parts[1])
    else:
        cats.add("(unparsed)")
for x in df["seed"]:
    seeds.add(int(x))
print(f"[reproduce_sec3_padim] marginal_protocol_b.csv: classes_count={len(cats)} seeds_count={len(seeds)} total_runs={len(df)}")
PY
elif [ -f "$STUB" ]; then
  n_data=$(($(wc -l < "$STUB") - 1))
  echo "[reproduce_sec3_padim] marginal_stub.csv only: total_rows(data)=$n_data (no full marginal_protocol_b.csv)"
else
  echo "[reproduce_sec3_padim] no marginal_protocol_b.csv or marginal_stub.csv yet — fast path will copy bundled stub"
fi

bash src/experiments/sec3_padim_observation/run.sh
