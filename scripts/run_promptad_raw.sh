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
#   PROMPTAD_CLASSES       — comma class names (required for train/infer), or **all** / **ALL**
#                          — expand to upstream lists per dataset (see external/PromptAD/datasets/mvtec.py, visa.py)
#   PROMPTAD_SHOTS         — comma ints (e.g. 1,2,4) for train/infer + export filter
#   PROMPTAD_SEEDS         — comma ints for train/infer + export filter
#   PROMPTAD_GPU           — CUDA_VISIBLE_DEVICES (default 0); train/test pass --gpu-id 0 with this mapping
#   PROMPTAD_EXTRA_ARGS    — extra CLI tokens for train_cls.py only (space-separated; not passed to test_cls.py)
#   PROMPTAD_INFER_EXTRA_ARGS — optional extra tokens for test_cls.py only
#
# Large-grid safety (train/infer loops only):
#   PROMPTAD_RESUME=1              — skip train and infer when CLS-*-per_sample.csv already exists for that cell
#   PROMPTAD_FAIL_FAST=1           — stop the grid on first train/infer failure (default: continue, still export at end)
#   PROMPTAD_STATUS_DIR            — default: $REPO_ROOT/outputs/promptad_fullpath_status
#   PROMPTAD_RUN_STATUS_CSV        — default: $PROMPTAD_STATUS_DIR/promptad_run_status.csv
#   PROMPTAD_EXIT_OK_ON_PARTIAL=1 — exit 0 even if some train/infer cells failed (default: exit 1 if any cell failed)
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
STATUS_PY="$REPO_ROOT/scripts/promptad_run_status.py"
STATUS_DIR="${PROMPTAD_STATUS_DIR:-$REPO_ROOT/outputs/promptad_fullpath_status}"
STATUS_CSV="${PROMPTAD_RUN_STATUS_CSV:-$STATUS_DIR/promptad_run_status.csv}"

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

has_train=0
has_infer=0
has_export=0
for m in "${MODES[@]}"; do
  case "$m" in
    train) has_train=1 ;;
    infer) has_infer=1 ;;
    export) has_export=1 ;;
  esac
done

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
  : "${PROMPTAD_CLASSES:?Set PROMPTAD_CLASSES (comma-separated or all) for train/infer}"
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

# Must match external/PromptAD/datasets/mvtec.py / visa.py (Stage 1 wrapper only; no training-code edits).
_PROMPTAD_MVTEC_ALL="carpet,grid,leather,tile,wood,bottle,cable,capsule,hazelnut,metal_nut,pill,screw,toothbrush,transistor,zipper"
_PROMPTAD_VISA_ALL="candle,capsules,cashew,chewinggum,fryum,macaroni1,macaroni2,pcb1,pcb2,pcb3,pcb4,pipe_fryum"
ALL_CLASSES_MODE=0
if [ -n "${PROMPTAD_CLASSES:-}" ]; then
  IFS=',' read -r -a _probe_cls <<< "${PROMPTAD_CLASSES// /}"
  if [ "${#_probe_cls[@]}" -eq 1 ]; then
    _cx="$(echo "${_probe_cls[0]}" | tr '[:upper:]' '[:lower:]' | xargs)"
    if [ "$_cx" = "all" ]; then
      ALL_CLASSES_MODE=1
    fi
  fi
fi

promptad_per_sample_relpath() {
  local ds="$1" class="$2" shot="$3" seed="$4"
  echo "${ds}/k_${shot}/csv/CLS-${ds}-${class}-k${shot}-seed${seed}-per_sample.csv"
}

promptad_status_append() {
  local ds="$1" class="$2" shot="$3" seed="$4" ts="$5" is="$6" es="$7" path="$8" st="$9" en="${10}" err="${11}"
  local errf=""
  local -a ef_args=()
  if [ -n "$err" ]; then
    errf="$(mktemp)"
    printf '%s' "$err" >"$errf"
    ef_args=(--error-file "$errf")
  fi
  "$PY" "$STATUS_PY" append "$STATUS_CSV" \
    --dataset "$ds" --category "$class" --shot "$shot" --seed "$seed" \
    --train-status "$ts" --infer-status "$is" \
    --export-status "$es" \
    --per-sample-path "$path" \
    --error-message "" \
    "${ef_args[@]}" \
    --start-time "$st" --end-time "$en"
  if [ -n "$errf" ] && [ -f "$errf" ]; then
    rm -f "$errf"
  fi
  return 0
}

if [ "$run_train_infer" = "1" ]; then
  echo "[run_promptad_raw] NOTE: upstream PromptAD loads MVTec/VisA from fixed paths (e.g. ~/datasets/mvtec)." >&2
  echo "       Ensure data are visible there, or symlink, e.g.: ln -sfn \"\$PROMPTAD_DATA_ROOT\" \"\$HOME/datasets/mvtec\" for MVTec layout." >&2
  IFS=',' read -r -a DATASETS <<< "${PROMPTAD_DATASETS// /}"
  IFS=',' read -r -a CLASSES <<< "${PROMPTAD_CLASSES// /}"
  IFS=',' read -r -a SHOTS <<< "${PROMPTAD_SHOTS// /}"
  IFS=',' read -r -a SEEDS <<< "${PROMPTAD_SEEDS// /}"

  if [ "$ALL_CLASSES_MODE" = "1" ]; then
    echo "[run_promptad_raw] PROMPTAD_CLASSES=all — per-dataset lists from PromptAD datasets/*.py (15 mvtec + 12 visa)." >&2
  fi

  mkdir -p "$STATUS_DIR"
  "$PY" "$STATUS_PY" init "$STATUS_CSV"
  echo "[run_promptad_raw] status CSV: $STATUS_CSV (PROMPTAD_RESUME=${PROMPTAD_RESUME:-0} PROMPTAD_FAIL_FAST=${PROMPTAD_FAIL_FAST:-0})" >&2

  any_cell_failure=0

  for ds in "${DATASETS[@]}"; do
    [ -n "$ds" ] || continue
    ds_lc="$(echo "$ds" | tr '[:upper:]' '[:lower:]')"
    if [ "$ALL_CLASSES_MODE" = "1" ]; then
      case "$ds_lc" in
        mvtec) IFS=',' read -r -a EFFECTIVE_CLASSES <<< "${_PROMPTAD_MVTEC_ALL// /}" ;;
        visa) IFS=',' read -r -a EFFECTIVE_CLASSES <<< "${_PROMPTAD_VISA_ALL// /}" ;;
        *)
          echo "ERROR: PROMPTAD_CLASSES=all is only defined for dataset mvtec or visa (got: $ds)" >&2
          exit 1
          ;;
      esac
    else
      EFFECTIVE_CLASSES=("${CLASSES[@]}")
    fi
    for class in "${EFFECTIVE_CLASSES[@]}"; do
      [ -n "$class" ] || continue
      for shot in "${SHOTS[@]}"; do
        [ -n "$shot" ] || continue
        for seed in "${SEEDS[@]}"; do
          [ -n "$seed" ] || continue

          rel="$(promptad_per_sample_relpath "$ds" "$class" "$shot" "$seed")"
          per_sample_host="${PROMPTAD_OUTPUT_ROOT%/}/$rel"
          per_sample_abs="$(realpath -m "$per_sample_host")"

          cell_start="$(date -Iseconds)"
          train_status="n/a"
          infer_status="n/a"
          err_msg=""
          export_status_row="n/a"
          if [ "$has_export" = "1" ]; then
            export_status_row="pending"
          fi

          skip_train_infer=0
          if [ "${PROMPTAD_RESUME:-0}" = "1" ] && [ -f "$per_sample_host" ]; then
            skip_train_infer=1
            if [ "$has_train" = "1" ]; then
              train_status="skipped_existing"
            fi
            if [ "$has_infer" = "1" ]; then
              infer_status="skipped_existing"
            fi
          fi

          if [ "$skip_train_infer" = "1" ]; then
            cell_end="$(date -Iseconds)"
            promptad_status_append "$ds" "$class" "$shot" "$seed" "$train_status" "$infer_status" \
              "$export_status_row" "$per_sample_abs" "$cell_start" "$cell_end" ""
            continue
          fi

          if [ "$has_train" = "1" ]; then
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
            tf="$(mktemp)"
            set +e
            "${CMD[@]}" >"$tf" 2>&1
            rc=$?
            set -euo pipefail
            if [ "$rc" -eq 0 ]; then
              train_status="ok"
            else
              train_status="failed"
              any_cell_failure=1
              err_msg="$(tail -c 1800 "$tf" | tr '\n\r' '  ')"
              if [ "${PROMPTAD_FAIL_FAST:-0}" = "1" ]; then
                rm -f "$tf"
                infer_status="skipped_fail_fast"
                cell_end="$(date -Iseconds)"
                promptad_status_append "$ds" "$class" "$shot" "$seed" "$train_status" "$infer_status" \
                  "$export_status_row" "$per_sample_abs" "$cell_start" "$cell_end" "$err_msg"
                echo "[run_promptad_raw] FAIL_FAST: stopping after train failure." >&2
                exit 1
              fi
            fi
            rm -f "${tf:-}"
          fi

          if [ "$has_infer" = "1" ]; then
            if [ "$has_train" = "1" ] && [ "$train_status" = "failed" ]; then
              infer_status="skipped_train_failed"
            else
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
              tf="$(mktemp)"
              set +e
              "${CMD[@]}" >"$tf" 2>&1
              rc=$?
              set -euo pipefail
              if [ "$rc" -eq 0 ]; then
                infer_status="ok"
              else
                infer_status="failed"
                any_cell_failure=1
                err_msg="$(tail -c 1800 "$tf" | tr '\n\r' '  ')"
                if [ "${PROMPTAD_FAIL_FAST:-0}" = "1" ]; then
                  rm -f "$tf"
                  cell_end="$(date -Iseconds)"
                  promptad_status_append "$ds" "$class" "$shot" "$seed" "$train_status" "$infer_status" \
                    "$export_status_row" "$per_sample_abs" "$cell_start" "$cell_end" "$err_msg"
                  echo "[run_promptad_raw] FAIL_FAST: stopping after infer failure." >&2
                  exit 1
                fi
              fi
              rm -f "${tf:-}"
            fi
          fi

          cell_end="$(date -Iseconds)"
          promptad_status_append "$ds" "$class" "$shot" "$seed" "$train_status" "$infer_status" \
            "$export_status_row" "$per_sample_abs" "$cell_start" "$cell_end" "$err_msg"
        done
      done
    done
  done

  export_failed=0
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
      if [ "${ALL_CLASSES_MODE:-0}" != "1" ] && [ -n "${PROMPTAD_CLASSES:-}" ]; then
        EXP_CMD+=(--classes-filter "$PROMPTAD_CLASSES")
      fi
      [ -n "${PROMPTAD_SHOTS:-}" ] && EXP_CMD+=(--shots-filter "$PROMPTAD_SHOTS")
      [ -n "${PROMPTAD_SEEDS:-}" ] && EXP_CMD+=(--seeds-filter "$PROMPTAD_SEEDS")
      set +e
      "${EXP_CMD[@]}"
      erc=$?
      set -euo pipefail
      if [ "$erc" -ne 0 ]; then
        export_failed=1
        echo "[run_promptad_raw] ERROR: export exited $erc" >&2
      else
        if [ -f "$STATUS_CSV" ]; then
          "$PY" "$STATUS_PY" finalize-export "$STATUS_CSV"
        fi
      fi
    fi
  done

  if [ "$export_failed" = "1" ]; then
    exit 1
  fi
  if [ "$any_cell_failure" = "1" ] && [ "${PROMPTAD_EXIT_OK_ON_PARTIAL:-0}" != "1" ]; then
    echo "[run_promptad_raw] WARNING: one or more train/infer cells failed (see $STATUS_CSV). Exiting 1." >&2
    exit 1
  fi
else
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
      if [ "${ALL_CLASSES_MODE:-0}" != "1" ] && [ -n "${PROMPTAD_CLASSES:-}" ]; then
        EXP_CMD+=(--classes-filter "$PROMPTAD_CLASSES")
      fi
      [ -n "${PROMPTAD_SHOTS:-}" ] && EXP_CMD+=(--shots-filter "$PROMPTAD_SHOTS")
      [ -n "${PROMPTAD_SEEDS:-}" ] && EXP_CMD+=(--seeds-filter "$PROMPTAD_SEEDS")
      "${EXP_CMD[@]}"
    fi
  done
fi

echo "[run_promptad_raw] done — raw evidence: $RAW_OUT"
