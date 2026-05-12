#!/usr/bin/env bash
# Stage 1 — PromptAD model evidence (train/infer optional; export from existing CSVs is default).
# Not invoked by reproduce_*.
#
# Required:
#   FULL_RUN=1
#   PROMPTAD_OUTPUT_ROOT   — where PromptAD writes results (train_cls/test_cls --root-dir) OR tree to scan for export
#
# Optional / mode-dependent:
#   PROMPTAD_DATA_ROOT     — required if PROMPTAD_MODE includes train or infer (dataset parent; see docs/FULLPATH_PROMPTAD.md)
#   PROMPTAD_RAW_OUT       — default: $REPO_ROOT/outputs/cached_results/raw_scores/promptad
#   PROMPTAD_MODE          — comma tokens: export (default), train, infer  (e.g. export | infer,export | train,infer,export)
#   PROMPTAD_DATASETS      — comma filter (e.g. mvtec) for export + loop domains for train/infer
#   PROMPTAD_CLASSES       — comma class names (required for train/infer)
#   PROMPTAD_SHOTS         — comma ints (e.g. 1,2,4) for train/infer + export filter
#   PROMPTAD_SEEDS         — comma ints for train/infer + export filter
#   PROMPTAD_GPU           — CUDA_VISIBLE_DEVICES (default 0); train/test pass --gpu-id 0 with this mapping
#   PROMPTAD_EXTRA_ARGS    — extra CLI tokens for train_cls.py only (space-separated; not passed to test_cls.py)
#   PROMPTAD_INFER_EXTRA_ARGS — optional extra tokens for test_cls.py only
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

if [ "${FULL_RUN:-0}" != "1" ]; then
  echo "ERROR: run_promptad_raw.sh runs only when FULL_RUN=1 (Stage 1 model evidence)." >&2
  exit 1
fi

: "${PROMPTAD_OUTPUT_ROOT:?Set PROMPTAD_OUTPUT_ROOT (PromptAD --root-dir target or tree to export)}"

RAW_OUT="${PROMPTAD_RAW_OUT:-$REPO_ROOT/outputs/cached_results/raw_scores/promptad}"
MODE="${PROMPTAD_MODE:-export}"
GPU="${PROMPTAD_GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
PY="${PYTHON:-python3}"

IFS=',' read -r -a MODE_ARR <<< "${MODE// /}"
MODES=()
for m in "${MODE_ARR[@]}"; do
  [ -n "$m" ] || continue
  mm="$(echo "$m" | tr '[:upper:]' '[:lower:]')"
  case "$mm" in
    export|train|infer) MODES+=("$mm") ;;
    *)
      echo "ERROR: unknown PROMPTAD_MODE token: $m (use export, train, infer)" >&2
      exit 1
      ;;
  esac
done
if [ "${#MODES[@]}" -eq 0 ]; then
  echo "ERROR: PROMPTAD_MODE empty after parse" >&2
  exit 1
fi

need_data=0
for m in "${MODES[@]}"; do
  if [ "$m" = "train" ] || [ "$m" = "infer" ]; then
    need_data=1
    break
  fi
done
if [ "$need_data" = "1" ]; then
  : "${PROMPTAD_DATA_ROOT:?Set PROMPTAD_DATA_ROOT when PROMPTAD_MODE includes train or infer}"
  if [ ! -d "$PROMPTAD_DATA_ROOT" ]; then
    echo "ERROR: PROMPTAD_DATA_ROOT is not a directory: $PROMPTAD_DATA_ROOT" >&2
    exit 1
  fi
  : "${PROMPTAD_CLASSES:?Set PROMPTAD_CLASSES (comma-separated) for train/infer}"
  : "${PROMPTAD_DATASETS:?Set PROMPTAD_DATASETS (comma-separated mvtec,visa) for train/infer}"
  : "${PROMPTAD_SHOTS:?Set PROMPTAD_SHOTS (comma-separated ints) for train/infer}"
  : "${PROMPTAD_SEEDS:?Set PROMPTAD_SEEDS (comma-separated ints) for train/infer}"
fi

run_train_infer=0
for m in "${MODES[@]}"; do
  if [ "$m" = "train" ] || [ "$m" = "infer" ]; then
    run_train_infer=1
    break
  fi
done

if [ "$run_train_infer" = "1" ]; then
  echo "[run_promptad_raw] NOTE: upstream PromptAD loads MVTec/VisA from fixed paths (e.g. ~/datasets/mvtec)." >&2
  echo "       Ensure data are visible there, or symlink, e.g.: ln -sfn \"\$PROMPTAD_DATA_ROOT\" \"\$HOME/datasets/mvtec\" for MVTec layout." >&2
  IFS=',' read -r -a DATASETS <<< "${PROMPTAD_DATASETS// /}"
  IFS=',' read -r -a CLASSES <<< "${PROMPTAD_CLASSES// /}"
  IFS=',' read -r -a SHOTS <<< "${PROMPTAD_SHOTS// /}"
  IFS=',' read -r -a SEEDS <<< "${PROMPTAD_SEEDS// /}"
  for ds in "${DATASETS[@]}"; do
    [ -n "$ds" ] || continue
    for class in "${CLASSES[@]}"; do
      [ -n "$class" ] || continue
      for shot in "${SHOTS[@]}"; do
        [ -n "$shot" ] || continue
        for seed in "${SEEDS[@]}"; do
          [ -n "$seed" ] || continue
          for m in "${MODES[@]}"; do
            if [ "$m" = "train" ]; then
              echo "[run_promptad_raw] train $ds $class k=$shot seed=$seed"
              CMD=(
                "$PY" -m src.models.promptad_adapter.run_promptad train_cls.py
                --dataset "$ds" --class_name "$class" --k-shot "$shot" --seed "$seed"
                --root-dir "$PROMPTAD_OUTPUT_ROOT" --gpu-id 0
              )
              if [ -n "${PROMPTAD_TRAIN_EXTRA_ARGS:-}" ]; then
                # shellcheck disable=SC2206
                CMD+=($PROMPTAD_TRAIN_EXTRA_ARGS)
              elif [ -n "${PROMPTAD_EXTRA_ARGS:-}" ]; then
                # shellcheck disable=SC2206
                CMD+=($PROMPTAD_EXTRA_ARGS)
              fi
              "${CMD[@]}"
            elif [ "$m" = "infer" ]; then
              echo "[run_promptad_raw] infer $ds $class k=$shot seed=$seed"
              CMD=(
                "$PY" -m src.models.promptad_adapter.run_promptad test_cls.py
                --dataset "$ds" --class_name "$class" --k-shot "$shot" --seed "$seed"
                --root-dir "$PROMPTAD_OUTPUT_ROOT" --gpu-id 0 --vis False
              )
              if [ -n "${PROMPTAD_INFER_EXTRA_ARGS:-}" ]; then
                # shellcheck disable=SC2206
                CMD+=($PROMPTAD_INFER_EXTRA_ARGS)
              fi
              "${CMD[@]}"
            fi
          done
        done
      done
    done
  done
fi

for m in "${MODES[@]}"; do
  if [ "$m" = "export" ]; then
    echo "[run_promptad_raw] export unified raw from $PROMPTAD_OUTPUT_ROOT -> $RAW_OUT"
    mkdir -p "$RAW_OUT"
    EXP_CMD=(
      "$PY" "$REPO_ROOT/src/models/promptad_adapter/promptad_export_unified_raw.py"
      --input-root "$PROMPTAD_OUTPUT_ROOT"
      --out-dir "$RAW_OUT"
    )
    [ -n "${PROMPTAD_DATASETS:-}" ] && EXP_CMD+=(--datasets-filter "$PROMPTAD_DATASETS")
    [ -n "${PROMPTAD_CLASSES:-}" ] && EXP_CMD+=(--classes-filter "$PROMPTAD_CLASSES")
    [ -n "${PROMPTAD_SHOTS:-}" ] && EXP_CMD+=(--shots-filter "$PROMPTAD_SHOTS")
    [ -n "${PROMPTAD_SEEDS:-}" ] && EXP_CMD+=(--seeds-filter "$PROMPTAD_SEEDS")
    "${EXP_CMD[@]}"
  fi
done

echo "[run_promptad_raw] done — raw evidence: $RAW_OUT"
