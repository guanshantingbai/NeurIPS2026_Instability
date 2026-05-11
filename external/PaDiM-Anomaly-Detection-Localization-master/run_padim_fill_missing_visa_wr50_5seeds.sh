#!/usr/bin/env bash
# Fill missing / incomplete VisA + wide_resnet50_2 rows under padim_result_seed_search_full:
#   - 11 slugs absent on disk (see run_padim_full_5seeds_allsettings.sh)
#   - visa__candle__wide_resnet50_2 (empty seed dirs / no summary.json)
#
# Resume: skips when summary.json exists for that (slug, seed).
# Matches run_padim_full_5seeds_allsettings.sh flags (incl. --cov-float32 for visa+WR50).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-/home/zju/miniconda3/envs/myenv/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/padim_result_seed_search_full}"
VISA_PATH="${VISA_PATH:-$HOME/datasets/pro_visa}"
SEEDS_CSV="${SEEDS:-111,222,333,444,555}"
# WR50 + full VisA test RAM: default batch 1; raise only if you have headroom.
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CPU_THREADS="${CPU_THREADS:-1}"
# Train Gaussian: VisA has 1000 train imgs × WR50 features → huge CPU RAM if uncapped.
# Match ``run_remaining_visa_wr50.sh`` / panel (a) default; set empty or 0 to disable cap (needs large RAM).
MAX_TRAIN_IMAGES="${MAX_TRAIN_IMAGES:-128}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Force streaming path for Protocol B test pass (avoids torch.cat of all test features on CPU for WR50).
export PADIM_STREAM_TEST_THRESHOLD="${PADIM_STREAM_TEST_THRESHOLD:-1}"

# Only classes that were still missing or broken in OUT_ROOT (VisA WR50).
VISA_WR50_FILL_CLASSES=(
  candle
  capsules
  cashew
  chewinggum
  fryum
  macaroni1
  macaroni2
  pcb1
  pcb2
  pcb3
  pcb4
  pipe_fryum
)

mkdir -p "$OUT_ROOT"

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export TORCH_NUM_THREADS="$CPU_THREADS"

IFS=',' read -r -a SEEDS_ARR <<< "$SEEDS_CSV"
ARCH="wide_resnet50_2"

run_one() {
  local cls="$1"
  local seed="$2"
  local slug="visa__${cls}__${ARCH}"
  local save_dir="$OUT_ROOT/$slug/$seed"
  local summary="$save_dir/summary.json"

  if [[ -f "$summary" ]]; then
    echo "[skip] $slug seed=$seed (summary exists)"
    return 0
  fi

  mkdir -p "$save_dir"
  echo "[run]  $slug seed=$seed"
  local cap=()
  if [[ -n "${MAX_TRAIN_IMAGES:-}" && "${MAX_TRAIN_IMAGES}" != "0" ]]; then
    cap+=(--max-train-images "$MAX_TRAIN_IMAGES")
  fi
  "$PY" "$ROOT/padim_protocol_b_one_run.py" \
    --dataset visa \
    --class_name "$cls" \
    --arch "$ARCH" \
    --seed "$seed" \
    --save_dir "$save_dir" \
    --data_path "$VISA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    "${cap[@]}" \
    --cov-float32
}

echo "OUT_ROOT=$OUT_ROOT"
echo "VISA_PATH=$VISA_PATH"
echo "Seeds: $SEEDS_CSV"
echo "MAX_TRAIN_IMAGES=$MAX_TRAIN_IMAGES (set MAX_TRAIN_IMAGES=0 for full train, needs high RAM)"
echo "PADIM_STREAM_TEST_THRESHOLD=$PADIM_STREAM_TEST_THRESHOLD"
echo "Classes (${#VISA_WR50_FILL_CLASSES[@]}): ${VISA_WR50_FILL_CLASSES[*]}"

for cls in "${VISA_WR50_FILL_CLASSES[@]}"; do
  for seed in "${SEEDS_ARR[@]}"; do
    run_one "$cls" "$seed"
  done
done

echo "Done. VisA WR50 fill sweep finished."
