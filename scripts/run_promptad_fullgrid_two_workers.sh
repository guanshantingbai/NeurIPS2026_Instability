#!/usr/bin/env bash
# Orchestrate PromptAD full grid (405 cells): two parallel workers (mvtec / visa),
# independent status CSV + logs, shared PROMPTAD_OUTPUT_ROOT, then merged status + export.
# Does not modify PromptAD training code.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"

OUT="${PROMPTAD_OUTPUT_ROOT:-$REPO_ROOT/outputs/promptad_fullpath_27cls_3shots_5seeds}"
STATUS_BASE="${PROMPTAD_STATUS_BASE:-$REPO_ROOT/outputs/promptad_fullpath_status/fullgrid_27cls_3shots_5seeds}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP="$STATUS_BASE/backups/$TS"
mkdir -p "$BACKUP" "$STATUS_BASE"

echo "[fullgrid] backup docs + prior status CSVs -> $BACKUP"
cp -a "$REPO_ROOT/docs/END_TO_END_VALIDATION.md" "$BACKUP/" 2>/dev/null || true
cp -a "$REPO_ROOT/docs/REPRODUCIBILITY_STATUS.md" "$BACKUP/" 2>/dev/null || true
cp -a "$REPO_ROOT/docs/PROMPTAD_RESULT_RECOVERY.md" "$BACKUP/" 2>/dev/null || true
shopt -s nullglob
for f in "$REPO_ROOT/outputs/promptad_fullpath_status"/*.csv; do
  cp -a "$f" "$BACKUP/" || true
done
for f in "$STATUS_BASE"/status_*.csv; do
  cp -a "$f" "$BACKUP/" || true
done

echo "[fullgrid] verify ~/datasets/mvtec and ~/datasets/pro_visa"
test -d "$HOME/datasets/mvtec" || { echo "ERROR: missing ~/datasets/mvtec"; exit 1; }
test -d "$HOME/datasets/pro_visa" || { echo "ERROR: missing ~/datasets/pro_visa"; exit 1; }

echo "[fullgrid] verify 15 mvtec + 12 visa class dirs"
python3 <<'PY'
import os
MROOT = os.path.expanduser("~/datasets/mvtec")
VROOT = os.path.expanduser("~/datasets/pro_visa")
mvtec = "carpet,grid,leather,tile,wood,bottle,cable,capsule,hazelnut,metal_nut,pill,screw,toothbrush,transistor,zipper".split(",")
visa = "candle,capsules,cashew,chewinggum,fryum,macaroni1,macaroni2,pcb1,pcb2,pcb3,pcb4,pipe_fryum".split(",")
for root, classes, name in [(MROOT, mvtec, "mvtec"), (VROOT, visa, "visa")]:
    for c in classes:
        for sub in ("train", "test"):
            p = os.path.join(root, c, sub)
            if not os.path.isdir(p):
                raise SystemExit(f"missing {name}: {p}")
print("OK: all class train/test dirs exist")
PY

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT"
GPU_A="${FULLGRID_GPU_A:-0}"
GPU_B="${FULLGRID_GPU_B:-0}"

if [ -z "${PROMPTAD_TRAIN_EXTRA_ARGS:-}" ]; then
  export PROMPTAD_TRAIN_EXTRA_ARGS="--Epoch 3 --batch-size 8 --eval-freq 1 --num-workers 2"
fi

run_worker() {
  local label="$1" datasets="$2" cvd="$3" status_csv="$4" logfile="$5"
  (
    export CUDA_VISIBLE_DEVICES="$cvd"
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
    export TOKENIZERS_PARALLELISM=false
    cd "$REPO_ROOT"
    # shellcheck disable=SC1091
    source "$REPO_ROOT/scripts/repo_env.sh"
    FULL_RUN=1 \
      PROMPTAD_MODE=train,infer \
      PROMPTAD_OUTPUT_ROOT="$OUT" \
      PROMPTAD_DATA_ROOT="${PROMPTAD_DATA_ROOT:-/home/zju/datasets/mvtec}" \
      PROMPTAD_DATASETS="$datasets" \
      PROMPTAD_CLASSES=all \
      PROMPTAD_SHOTS=1,2,4 \
      PROMPTAD_SEEDS=111,222,333,444,555 \
      PROMPTAD_GPU=0 \
      PROMPTAD_TRAIN_EXTRA_ARGS="$PROMPTAD_TRAIN_EXTRA_ARGS" \
      PROMPTAD_RESUME=1 \
      PROMPTAD_FAIL_FAST=0 \
      PROMPTAD_EXIT_OK_ON_PARTIAL=1 \
      PROMPTAD_STATUS_DIR="$STATUS_BASE" \
      PROMPTAD_RUN_STATUS_CSV="$STATUS_BASE/$status_csv" \
      bash "$REPO_ROOT/scripts/run_promptad_raw.sh"
  ) >"$STATUS_BASE/$logfile" 2>&1
  echo $? >"$STATUS_BASE/${logfile%.log}.exit"
}

echo "[fullgrid] starting worker A (mvtec) GPU visible=$GPU_A -> $STATUS_BASE/status_worker_mvtec.csv"
run_worker A mvtec "$GPU_A" status_worker_mvtec.csv worker_mvtec.log &
PID_A=$!

echo "[fullgrid] starting worker B (visa) GPU visible=$GPU_B -> $STATUS_BASE/status_worker_visa.csv"
run_worker B visa "$GPU_B" status_worker_visa.csv worker_visa.log &
PID_B=$!

wait "$PID_A" || true
wait "$PID_B" || true
echo "[fullgrid] workers finished (see $STATUS_BASE/worker_*.exit)"

do_merge_export_summary() {
python3 "$REPO_ROOT/scripts/merge_promptad_status_csvs.py" \
  "$STATUS_BASE/status_worker_mvtec.csv" \
  "$STATUS_BASE/status_worker_visa.csv" \
  -o "$STATUS_BASE/status_merged.csv"

echo "[fullgrid] export unified raw"
FULL_RUN=1 \
  PROMPTAD_MODE=export \
  PROMPTAD_OUTPUT_ROOT="$OUT" \
  PROMPTAD_DATASETS=mvtec,visa \
  PROMPTAD_CLASSES=all \
  PROMPTAD_SHOTS=1,2,4 \
  PROMPTAD_SEEDS=111,222,333,444,555 \
  bash "$REPO_ROOT/scripts/run_promptad_raw.sh"

python3 "$REPO_ROOT/scripts/promptad_run_status.py" finalize-export "$STATUS_BASE/status_merged.csv" || true

echo "[fullgrid] summary counts"
export REPO_ROOT OUT STATUS_BASE
python3 <<'PY'
import csv, os
from pathlib import Path
repo = Path(os.environ["REPO_ROOT"])
out = Path(os.environ["OUT"])
status = Path(os.environ["STATUS_BASE"]) / "status_merged.csv"
rawd = repo / "outputs/cached_results/raw_scores/promptad"
per = [p for p in out.rglob("CLS-*-per_sample.csv") if "instability" not in p.name and "fusion" not in p.name]
print("per_sample_csv", len(per))
for name in ("unified_raw_scores_wide.csv", "unified_raw_scores_long.csv", "manifest.json"):
    p = rawd / name
    print(name, "exists" if p.is_file() else "MISSING")
if (rawd / "unified_raw_scores_wide.csv").is_file():
    n = sum(1 for _ in open(rawd / "unified_raw_scores_wide.csv", encoding="utf-8", errors="replace")) - 1
    print("wide_data_rows", n)
if (rawd / "unified_raw_scores_long.csv").is_file():
    n = sum(1 for _ in open(rawd / "unified_raw_scores_long.csv", encoding="utf-8", errors="replace")) - 1
    print("long_data_rows", n)
rows = list(csv.DictReader(open(status, newline="", encoding="utf-8")))
print("status_merged_rows", len(rows))
fail = [r for r in rows if r.get("train_status") == "failed" or r.get("infer_status") == "failed"]
print("failed_cells", len(fail))
fb = Path(os.environ["STATUS_BASE"]) / "failed_cells.txt"
with fb.open("w", encoding="utf-8") as f:
    for r in fail:
        f.write(
            f"{r.get('dataset')}\t{r.get('category')}\t{r.get('shot')}\t{r.get('seed')}\t{r.get('train_status')}\t{r.get('infer_status')}\t{r.get('error_message','')[:200]}\n"
        )
print("wrote", fb)
for r in fail[:50]:
    print("FAIL", r.get("dataset"), r.get("category"), r.get("shot"), r.get("seed"), r.get("train_status"), r.get("infer_status"))
if len(fail) > 50:
    print("... and", len(fail) - 50, "more")
(Path(os.environ["STATUS_BASE"]) / "failed_count.txt").write_text(str(len(fail)), encoding="utf-8")
PY
}

do_merge_export_summary

FAILN="$(cat "$STATUS_BASE/failed_count.txt" 2>/dev/null || echo 0)"
if [ "${FAILN:-0}" -gt 0 ]; then
  echo "[fullgrid] $FAILN failed cells — retry workers with PROMPTAD_RESUME=1 (same logs overwritten)"
  run_worker A mvtec "$GPU_A" status_worker_mvtec.csv worker_mvtec.log &
  PID_A=$!
  run_worker B visa "$GPU_B" status_worker_visa.csv worker_visa.log &
  PID_B=$!
  wait "$PID_A" || true
  wait "$PID_B" || true
  do_merge_export_summary
fi

echo "[fullgrid] orchestrator done"
