#!/usr/bin/env bash
set -euo pipefail

# Full settings = (dataset, category, backbone) over MVTec+VisA and R18+WR50.
# Seeds default to 111,222,333,444,555.
# Uses Protocol B one-run script and writes one folder per (setting, seed).
#
# Resume behavior: if summary.json exists for a (setting, seed), it is skipped.

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-/home/zju/miniconda3/envs/myenv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/padim_result_seed_search_full}"
MVTec_PATH="${MVTec_PATH:-$HOME/datasets/mvtec}"
VISA_PATH="${VISA_PATH:-$HOME/datasets/pro_visa}"
SEEDS_CSV="${SEEDS:-111,222,333,444,555}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CPU_THREADS="${CPU_THREADS:-2}"

mkdir -p "$OUT_ROOT"

# Per-process CPU parallelism cap (BLAS/OpenMP/NumExpr).
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export TORCH_NUM_THREADS="$CPU_THREADS"

IFS=',' read -r -a SEEDS_ARR <<< "$SEEDS_CSV"
BACKBONES=("resnet18" "wide_resnet50_2")
MVTec_CLASSES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
VISA_CLASSES=(candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)

run_one() {
  local dataset="$1"
  local cls="$2"
  local arch="$3"
  local seed="$4"
  local data_path="$5"
  local slug="${dataset}__${cls}__${arch}"
  local save_dir="$OUT_ROOT/$slug/$seed"
  local summary="$save_dir/summary.json"

  if [[ -f "$summary" ]]; then
    echo "[skip] $slug seed=$seed (summary exists)"
    return 0
  fi

  mkdir -p "$save_dir"
  local extra=()
  if [[ "$dataset" == "visa" && "$arch" == "wide_resnet50_2" ]]; then
    extra+=(--cov-float32)
  fi

  echo "[run]  $slug seed=$seed"
  "$PY" "$ROOT/padim_protocol_b_one_run.py" \
    --dataset "$dataset" \
    --class_name "$cls" \
    --arch "$arch" \
    --seed "$seed" \
    --save_dir "$save_dir" \
    --data_path "$data_path" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    "${extra[@]}"
}

echo "Output root: $OUT_ROOT"
echo "Seeds: $SEEDS_CSV"
echo "Backbones: ${BACKBONES[*]}"
echo "CPU thread cap per process: $CPU_THREADS"

for arch in "${BACKBONES[@]}"; do
  for cls in "${MVTec_CLASSES[@]}"; do
    for seed in "${SEEDS_ARR[@]}"; do
      run_one "mvtec" "$cls" "$arch" "$seed" "$MVTec_PATH"
    done
  done
  for cls in "${VISA_CLASSES[@]}"; do
    for seed in "${SEEDS_ARR[@]}"; do
      run_one "visa" "$cls" "$arch" "$seed" "$VISA_PATH"
    done
  done
done

echo "Done. Full 5-seed sweep finished."
