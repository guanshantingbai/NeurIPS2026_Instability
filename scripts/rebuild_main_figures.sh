#!/usr/bin/env bash
# =============================================================================
# Stage 2 + 3 convenience entry — main-paper figures
#
#   Stage 2: aggregate CSV from Stage 1 unified_raw_scores (no train/infer)
#   Stage 3: paper figures under outputs/figures/{sec3_promptad,sec3_padim,section4}
#
# Does NOT run Stage 0 (preparation) or Stage 1 (model evidence).
# For appendix figures see scripts/rebuild_appendix_figures.sh (Stage 4, partial).
#
# Usage:
#   bash scripts/rebuild_main_figures.sh
#   FORCE_REBUILD=1 PROMPTAD_SAVE_PAIRWISE=1 PROMPTAD_PAIRWISE_WORKERS=8 bash scripts/rebuild_main_figures.sh
#
# Environment:
#   PROMPTAD_SAVE_PAIRWISE=1     write pairwise_metrics.csv (~15M rows, ~1.2GB)
#   PROMPTAD_PAIRWISE_WORKERS=8  Stage 2 vectorized parallelism
#   FORCE_REBUILD=1              ignore aggregation_done.json cache
#   SEC3_PROMPTAD_ALLOW_STUB=1   PromptAD stub fallback (no Stage 1 raw)
#
# Pipeline reference: docs/PIPELINE_STAGES.md
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/repo_env.sh"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PY="${PYTHON:-python3}"
RAW_DIR="${PROMPTAD_RAW_DIR:-$REPO_ROOT/outputs/cached_results/raw_scores/promptad}"
AGG_DIR="${PROMPTAD_AGGREGATE_DIR:-$REPO_ROOT/outputs/cached_results/promptad_pairwise}"
SEC3_CACHE="${SEC3_PROMPTAD_CACHE_DIR:-$REPO_ROOT/outputs/cached_results/sec3_promptad}"
SEC4_CACHE="${SEC4_SYSTEMATIC_CACHE_DIR:-$REPO_ROOT/outputs/cached_results/sec4_systematic}"
FIG_SEC3_PA="${FIG_SEC3_PROMPTAD_DIR:-$REPO_ROOT/outputs/figures/sec3_promptad}"
FIG_SEC3_PD="${FIG_SEC3_PADIM_DIR:-$REPO_ROOT/outputs/figures/sec3_padim}"
FIG_SEC4="${FIG_SECTION4_DIR:-$REPO_ROOT/outputs/figures/section4}"

export PROMPTAD_SAVE_PAIRWISE="${PROMPTAD_SAVE_PAIRWISE:-1}"
export PROMPTAD_PAIRWISE_WORKERS="${PROMPTAD_PAIRWISE_WORKERS:-8}"
WORKERS="${PROMPTAD_PAIRWISE_WORKERS}"
FORCE="${FORCE_REBUILD:-0}"

mkdir -p "$FIG_SEC3_PA" "$FIG_SEC3_PD" "$FIG_SEC4"
mkdir -p "$SEC3_CACHE" "$SEC4_CACHE" "$AGG_DIR"

WIDE="$RAW_DIR/unified_raw_scores_wide.csv"
LONG="$RAW_DIR/unified_raw_scores_long.csv"

echo "=============================================================================="
echo "[rebuild_main] Stage 2+3 — main-paper figures"
echo "  outputs/figures/{sec3_promptad,sec3_padim,section4}"
echo "  (requires Stage 1 unified raw; see docs/PIPELINE_STAGES.md)"
echo "=============================================================================="
echo "  repo root              : $REPO_ROOT"
echo "  PromptAD raw (wide)    : $WIDE"
echo "  PromptAD raw (long)    : $LONG"
echo "  Stage 2 aggregate dir  : $AGG_DIR"
echo "  sec3_promptad cache    : $SEC3_CACHE"
echo "  sec4_systematic cache  : $SEC4_CACHE"
echo "  PROMPTAD_SAVE_PAIRWISE : ${PROMPTAD_SAVE_PAIRWISE}"
echo "  PROMPTAD_PAIRWISE_WORKERS: ${WORKERS}"
echo "  FORCE_REBUILD          : ${FORCE}"
if [ "${PROMPTAD_SAVE_PAIRWISE}" = "1" ]; then
  echo "  WARNING: pairwise_metrics.csv can be large (~15M rows on full PromptAD grid)."
fi
echo "=============================================================================="

# -----------------------------------------------------------------------------
# Stage 2 — PromptAD pairwise aggregation (vectorized; mirrors sec3 + sec4 caches)
# -----------------------------------------------------------------------------
AGG_EXTRA=()
if [ -n "${PROMPTAD_MAX_PAIRS_PER_SETTING:-}" ]; then
  AGG_EXTRA+=(--max-pairs-per-setting "${PROMPTAD_MAX_PAIRS_PER_SETTING}")
fi
SAVE_PW=()
if [ "${PROMPTAD_SAVE_PAIRWISE}" = "1" ]; then
  SAVE_PW=(--save-pairwise)
fi

echo ""
echo "[rebuild_main] Stage 2/5: PromptAD pairwise aggregation (workers=$WORKERS)"
if [ "$FORCE" = "1" ] || [ ! -f "$AGG_DIR/aggregation_done.json" ]; then
  if [ ! -f "$WIDE" ] && [ ! -f "$LONG" ]; then
    echo "[rebuild_main] ERROR: missing PromptAD unified raw under $RAW_DIR" >&2
    echo "  Run Stage 1 first: FULL_RUN=1 bash scripts/run_promptad_raw.sh" >&2
    echo "  See docs/PIPELINE_STAGES.md and docs/FULLPATH_PROMPTAD.md" >&2
    exit 1
  fi
  "$PY" "$REPO_ROOT/scripts/run_promptad_pairwise_aggregation.py" \
    --raw-dir "$RAW_DIR" \
    --out-dir "$AGG_DIR" \
    --workers "$WORKERS" \
    --mirror-dirs "$SEC3_CACHE" "$SEC4_CACHE" \
    "${AGG_EXTRA[@]}" \
    "${SAVE_PW[@]}"
else
  echo "[rebuild_main]   Reusing $AGG_DIR/aggregation_done.json (FORCE_REBUILD=1 to rerun Stage 2)"
fi

# -----------------------------------------------------------------------------
# Stage 2 — PromptAD sec3 supplementary tables (risk_coverage, same-AUROC, …)
# -----------------------------------------------------------------------------
echo ""
echo "[rebuild_main] Stage 2/5: sec3_promptad aggregate CSVs (risk_coverage, same-AUROC, …)"
SEC3_EXTRA=()
if [ -n "${SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING:-}" ]; then
  SEC3_EXTRA+=(--max-pairs-per-setting "${SEC3_PROMPTAD_MAX_PAIRS_PER_SETTING}")
fi
if [ -n "${SEC3_PROMPTAD_PAIR_SAMPLING_SEED:-}" ]; then
  SEC3_EXTRA+=(--pair-sampling-seed "${SEC3_PROMPTAD_PAIR_SAMPLING_SEED}")
fi

SEC3_RC="$SEC3_CACHE/risk_coverage.csv"
SEC3_DONE="$FIG_SEC3_PA/from_raw_done.txt"
if [ "$FORCE" = "1" ] || [ ! -f "$SEC3_RC" ] || [ ! -f "$SEC3_DONE" ]; then
  if [ -f "$WIDE" ] || [ -f "$LONG" ]; then
    "$PY" "$REPO_ROOT/src/experiments/sec3_promptad_observation/analyze_sec3_promptad_from_raw.py" \
      --raw-dir "$RAW_DIR" \
      --cache-dir "$SEC3_CACHE" \
      --fig-dir "$FIG_SEC3_PA" \
      "${SEC3_EXTRA[@]}"
    echo "sec3_promptad Stage 2 aggregates from raw" >"$SEC3_DONE"
  elif [ "${SEC3_PROMPTAD_ALLOW_STUB:-0}" = "1" ]; then
    SAMPLE="$REPO_ROOT/src/experiments/sec3_promptad_observation/samples/fastpath"
    shopt -s nullglob
    stub_csv=("$SAMPLE/cached_results/"*.csv)
    cp -f "${stub_csv[@]}" "$SEC3_CACHE/"
    "$PY" "$REPO_ROOT/src/experiments/sec3_promptad_observation/plot_promptad_observation.py"
    echo "sec3_promptad stub fallback" >"$SEC3_DONE"
  else
    echo "[rebuild_main] ERROR: no PromptAD raw and SEC3_PROMPTAD_ALLOW_STUB!=1" >&2
    exit 1
  fi
else
  echo "[rebuild_main]   Reusing sec3_promptad Stage 2 caches ($SEC3_RC exists)"
fi

# -----------------------------------------------------------------------------
# Stage 3 — Main-paper figures (no new aggregation from unified raw)
# -----------------------------------------------------------------------------
echo ""
echo "[rebuild_main] Stage 3/5: sec3_promptad paper_style_fig1"
"$PY" "$REPO_ROOT/src/experiments/sec3_promptad_observation/build_paper_style_fig1.py" \
  --cache-dir "$SEC3_CACHE" \
  --fig-dir "$FIG_SEC3_PA"

echo ""
echo "[rebuild_main] Stage 3/5: section4 paper figures (fig2, fig3_4, fig5_6, fig7)"
if [ ! -f "$SEC3_CACHE/setting_level_metrics.csv" ]; then
  echo "[rebuild_main] ERROR: missing $SEC3_CACHE/setting_level_metrics.csv" >&2
  exit 1
fi
if [ ! -f "$SEC3_RC" ]; then
  echo "[rebuild_main] ERROR: missing $SEC3_RC (Section 4 risk @ coverage)" >&2
  exit 1
fi
if [ "${PROMPTAD_SAVE_PAIRWISE}" = "1" ] && [ ! -f "$SEC4_CACHE/pairwise_metrics.csv" ]; then
  echo "[rebuild_main] ERROR: PROMPTAD_SAVE_PAIRWISE=1 but $SEC4_CACHE/pairwise_metrics.csv missing" >&2
  exit 1
fi

"$PY" "$REPO_ROOT/src/experiments/sec4_systematic_validation/build_section4_paper_figures.py" \
  --repo-root "$REPO_ROOT" \
  --cache-dir "$SEC4_CACHE" \
  --sec3-cache-dir "$SEC3_CACHE" \
  --aggregate-dir "$AGG_DIR" \
  --fig-dir "$FIG_SEC4"

echo ""
echo "[rebuild_main] Stage 3/5: sec3_padim (Stage 2 marginal cache + paper_style_fig2)"
bash "$REPO_ROOT/src/experiments/sec3_padim_observation/run.sh"

FIG2_BUILDER="$REPO_ROOT/src/experiments/sec3_padim_observation/build_paper_style_fig2_padim.py"
if [ -f "$FIG2_BUILDER" ]; then
  "$PY" "$FIG2_BUILDER" \
    --cache-dir "$REPO_ROOT/outputs/cached_results/sec3_padim" \
    --fig-dir "$FIG_SEC3_PD"
else
  echo "[rebuild_main]   NOTE: $FIG2_BUILDER not found — skip paper_style_fig2 (diagnostics only)"
fi

# -----------------------------------------------------------------------------
echo ""
echo "[rebuild_main] done (Stage 2+3)."
echo "  sec3_promptad : $FIG_SEC3_PA"
echo "  sec3_padim    : $FIG_SEC3_PD"
echo "  section4      : $FIG_SEC4"
echo "  Stage 2 caches: $SEC3_CACHE, $SEC4_CACHE, $AGG_DIR"
if [ -f "$SEC4_CACHE/fig3_4_mechanism_summary.json" ]; then
  echo "  section4 (fig3_4 summary):"
  "$PY" -c "import json; d=json.load(open('$SEC4_CACHE/fig3_4_mechanism_summary.json')); print('    epsilon=',d.get('epsilon')); print('    tau=',d.get('tau')); print('    failure:', d.get('panel_c',{}).get('failure_definition',''))"
fi
