#!/usr/bin/env bash
# Stage 1 — PaDiM Protocol B raw scores (GPU + data). Not invoked by reproduce_*.
#
# Required:
#   FULL_RUN=1
#   PADIM_DATA_ROOT     — MVTec parent or VisA root (passed to padim_protocol_b_one_run --data_path)
#   PADIM_OUTPUT_ROOT   — writable root for per-job dirs (not git-tracked; use scratch or outputs/...)
#   PADIM_BACKBONES     — comma-separated: resnet18 and/or wide_resnet50_2
#   PADIM_SEEDS         — comma-separated integers (e.g. 444,555)
#
# Classes:
#   PADIM_CLASSES       — optional comma-separated class names. When **set** (non-empty), always wins over profile.
#   PADIM_PROFILE       — optional: `debug` | `paper` (default: **paper**). Used **only** when PADIM_CLASSES is unset/empty:
#       - paper + PADIM_DATASET=mvtec (default): auto-fill full MVTec-15 object class list.
#       - debug: auto-fill five-class smoke list (bottle,cable,capsule,screw,toothbrush).
#     For non-mvtec datasets, you must set PADIM_CLASSES explicitly.
#
# Optional:
#   PADIM_DATASET=mvtec|visa   (default mvtec)
#   PADIM_GPU=0                — sets CUDA_VISIBLE_DEVICES for upstream (cuda:0 inside visible set)
#   PADIM_FORCE=1              — re-run jobs even if per_sample.csv exists
#   PADIM_EXTRA_ARGS           — extra tokens for padim_protocol_b_one_run.py (quoted string split on spaces)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

if [ "${FULL_RUN:-0}" != "1" ]; then
  echo "ERROR: run_padim_raw.sh runs real PaDiM Protocol B only when FULL_RUN=1." >&2
  exit 1
fi

: "${PADIM_DATA_ROOT:?Set PADIM_DATA_ROOT}"
: "${PADIM_OUTPUT_ROOT:?Set PADIM_OUTPUT_ROOT}"
: "${PADIM_BACKBONES:?Set PADIM_BACKBONES (comma-separated resnet18 / wide_resnet50_2)}"
: "${PADIM_SEEDS:?Set PADIM_SEEDS (comma-separated ints)}"

if [ ! -d "$PADIM_DATA_ROOT" ]; then
  echo "ERROR: PADIM_DATA_ROOT is not a directory: $PADIM_DATA_ROOT" >&2
  exit 1
fi

DATASET="${PADIM_DATASET:-mvtec}"
PADIM_PROFILE="${PADIM_PROFILE:-paper}"

# --- PADIM_CLASSES: explicit wins; else profile-based defaults ---
if [ -n "${PADIM_CLASSES:-}" ]; then
  echo "[run_padim_raw] PADIM_PROFILE=$PADIM_PROFILE (explicit PADIM_CLASSES is set; profile does not override class list)"
else
  case "$PADIM_PROFILE" in
    paper)
      if [ "$DATASET" != "mvtec" ]; then
        echo "ERROR: PADIM_PROFILE=paper with empty PADIM_CLASSES requires PADIM_DATASET=mvtec (got: $DATASET). Set PADIM_CLASSES explicitly." >&2
        exit 1
      fi
      PADIM_CLASSES="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"
      export PADIM_CLASSES
      echo "[run_padim_raw] PADIM_PROFILE=paper: auto PADIM_CLASSES=MVTec-15 ($PADIM_CLASSES)"
      ;;
    debug)
      PADIM_CLASSES="bottle,cable,capsule,screw,toothbrush"
      export PADIM_CLASSES
      echo "[run_padim_raw] PADIM_PROFILE=debug: auto PADIM_CLASSES=smoke-5 ($PADIM_CLASSES)"
      ;;
    *)
      echo "ERROR: PADIM_PROFILE must be 'debug' or 'paper' (got: $PADIM_PROFILE)" >&2
      exit 1
      ;;
  esac
fi

GPU="${PADIM_GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

PY="${PYTHON:-python3}"
mkdir -p "${PADIM_OUTPUT_ROOT}/protocol_b_jobs"
if [[ "${PADIM_OUTPUT_ROOT}" = /* ]]; then
  JOBS_ROOT="$(cd "${PADIM_OUTPUT_ROOT}/protocol_b_jobs" && pwd)"
else
  JOBS_ROOT="$(cd "$REPO_ROOT/${PADIM_OUTPUT_ROOT}/protocol_b_jobs" && pwd)"
fi
RAW_OUT="${REPO_ROOT}/outputs/cached_results/raw_scores/padim"

IFS=',' read -r -a CLASSES <<< "${PADIM_CLASSES// /}"
IFS=',' read -r -a BACKBONES <<< "${PADIM_BACKBONES// /}"
IFS=',' read -r -a SEEDS <<< "${PADIM_SEEDS// /}"
n_class=0
for class in "${CLASSES[@]}"; do
  [ -n "$class" ] || continue
  n_class=$((n_class + 1))
done
n_seed=0
for seed in "${SEEDS[@]}"; do
  [ -n "$seed" ] || continue
  n_seed=$((n_seed + 1))
done
n_bb=0
for arch in "${BACKBONES[@]}"; do
  [ -n "$arch" ] || continue
  n_bb=$((n_bb + 1))
done
planned=$((n_class * n_bb * n_seed))

echo "[run_padim_raw] dataset=$DATASET PADIM_PROFILE=$PADIM_PROFILE jobs_root=$JOBS_ROOT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run_padim_raw] plan: classes=$n_class backbones=$n_bb seeds=$n_seed -> total_runs=$planned (Cartesian product)"

for class in "${CLASSES[@]}"; do
  [ -n "$class" ] || continue
  for arch in "${BACKBONES[@]}"; do
    [ -n "$arch" ] || continue
    slug="${DATASET}__${class}__${arch}"
    for seed in "${SEEDS[@]}"; do
      [ -n "$seed" ] || continue
      SAVE="${JOBS_ROOT}/${slug}/${seed}"
      mkdir -p "$SAVE"
      if [ -f "$SAVE/per_sample.csv" ] && [ "${PADIM_FORCE:-0}" != "1" ]; then
        echo "[run_padim_raw] skip existing $SAVE"
        continue
      fi
      echo "[run_padim_raw] run $slug seed=$seed -> $SAVE"
      CMD=(
        "$PY" -m src.models.padim_adapter.run_padim padim_protocol_b_one_run.py
        --dataset "$DATASET"
        --class_name "$class"
        --arch "$arch"
        --seed "$seed"
        --save_dir "$SAVE"
        --data_path "$PADIM_DATA_ROOT"
      )
      if [ -n "${PADIM_EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        CMD+=($PADIM_EXTRA_ARGS)
      fi
      "${CMD[@]}"
    done
  done
done

echo "[run_padim_raw] aggregating unified raw + cached tables"
"$PY" "$REPO_ROOT/src/models/padim_adapter/padim_aggregate_protocol_b_raw.py" \
  --jobs-root "$JOBS_ROOT" \
  --raw-out "$RAW_OUT" \
  --sec3-out "$REPO_ROOT/outputs/cached_results/sec3_padim" \
  --appendix-e-out "$REPO_ROOT/outputs/cached_results/app_padim_representation"

echo "[run_padim_raw] done — unified scores: $RAW_OUT"
