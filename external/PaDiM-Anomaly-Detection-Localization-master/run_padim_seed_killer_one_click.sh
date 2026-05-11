#!/usr/bin/env bash
# =============================================================================
# PaDiM「same-setting, different seeds」killer evidence — 一键流水线
#
# 逻辑（与 PromptAD seed killer 同构）：
#   1) 筛选（phase1）— 用 multiclass Protocol B 的 metrics 给每类打分，取 Top-K
#   2) 「训练/推理」（phase2）— 对每个 (类×arch×seed) 跑 padim_protocol_b_one_run
#      （PaDiM 无 k-shot；这里是 Protocol B 全量推理 + 高斯，不是 main.py 的 pkl 缓存链）
#   3) 再筛选 + 制图（phase3）— 在 |ΔAUROC| 小的 seed 对里按 risk–coverage 等选 killer，写 PNG/JSON
#   4) 附录图 — 排除主 killer 后再画若干张 risk–coverage
#
# 环境变量（可选）：
#   MVTec_PATH   默认 ~/datasets/mvtec
#   VISA_PATH    默认 ~/datasets/pro_visa（仅 VisA 类列表 phase1 时用）
#   SEEDS        默认 111,222,333,444,555
#   TOP_N        默认 8
#   PROXY        默认 u6（可选 u2、padim_marg）
#   METRICS_JSON phase1 输入，默认本仓库 protocol_b_mvtec6_r18/...
#   SKIP_PHASE2  设为 1 则跳过已耗时的多 seed 推理（仅当你已跑完 phase2）
#   SKIP_APPENDIX 设为 1 则跳过附录图
#   更强「killer」叙事（AUROC 更贴 + risk 曲线一条严格优于另一条）示例：
#     PHASE3_DELTA_AUROC_MAX=0.001 PHASE3_KILLER_SORT=risk bash ...
#     或 PHASE3_KILLER_SORT=dominance_first bash ...
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PADIM_ROOT="$SCRIPT_DIR"
REPO_ROOT="$(cd "${PADIM_ROOT}/.." && pwd)"

export MVTec_PATH="${MVTec_PATH:-$HOME/datasets/mvtec}"
export VISA_PATH="${VISA_PATH:-$HOME/datasets/pro_visa}"
SEEDS="${SEEDS:-111,222,333,444,555}"
TOP_N="${TOP_N:-8}"
PROXY="${PROXY:-u6}"
# Phase3: tighter AUROC (e.g. 0.001) + dominance_first → stronger nested risk–coverage (often bottle).
PHASE3_DELTA_AUROC_MAX="${PHASE3_DELTA_AUROC_MAX:-0.01}"
PHASE3_TOP_PAIRS="${PHASE3_TOP_PAIRS:-0}"
PHASE3_KILLER_SORT="${PHASE3_KILLER_SORT:-risk}"
METRICS_JSON="${METRICS_JSON:-${PADIM_ROOT}/protocol_b_mvtec6_r18/protocol_b_multiclass_metrics.json}"

SEARCH_ROOT="${PADIM_ROOT}/padim_result_seed_search"
PHASE1_DIR="${SEARCH_ROOT}/phase1"
PHASE2_SH="${SEARCH_ROOT}/phase2_protocol_b.sh"
ANALYSIS_DIR="${SEARCH_ROOT}/analysis"
PY="${PYTHON:-python3}"
PIPE="${PADIM_ROOT}/padim_seed_killer_evidence_pipeline.py"
APPENDIX="${PADIM_ROOT}/padim_seed_killer_appendix_pair_figures.py"

echo "== [1/4] Phase1: 筛选 Top-${TOP_N} settings → ${PHASE1_DIR}"
"${PY}" "${PIPE}" phase1 \
  --metrics-json "${METRICS_JSON}" \
  --dataset mvtec \
  --top-n "${TOP_N}" \
  --out-dir "${PHASE1_DIR}"

echo "== [2/4] Phase2: 生成多 seed 任务脚本 → ${PHASE2_SH}"
"${PY}" "${PIPE}" phase2 \
  --top-json "${PHASE1_DIR}/top_settings.json" \
  --seeds "${SEEDS}" \
  --search-root "${SEARCH_ROOT}" \
  --out-sh "${PHASE2_SH}" \
  --data-path-mvtec "${MVTec_PATH}" \
  --data-path-visa "${VISA_PATH}"

if [[ "${SKIP_PHASE2:-0}" != "1" ]]; then
  echo "== [2/4] 执行 Protocol B 多 seed 推理（耗时；可 Ctrl+C 后设 SKIP_PHASE2=1 仅跑后续）"
  cd "${PADIM_ROOT}"
  bash "${PHASE2_SH}"
else
  echo "== [2/4] SKIP_PHASE2=1，跳过推理"
fi

echo "== [3/4] Phase3: seed 对再筛选 + killer_final.png (ΔAUROC<=${PHASE3_DELTA_AUROC_MAX}, killer_sort=${PHASE3_KILLER_SORT})"
"${PY}" "${PIPE}" phase3 \
  --top-json "${PHASE1_DIR}/top_settings.json" \
  --search-root "${SEARCH_ROOT}" \
  --seeds "${SEEDS}" \
  --proxy "${PROXY}" \
  --delta-auroc-max "${PHASE3_DELTA_AUROC_MAX}" \
  --top-pairs-per-setting "${PHASE3_TOP_PAIRS}" \
  --killer-sort "${PHASE3_KILLER_SORT}" \
  --out-analysis "${ANALYSIS_DIR}"

if [[ "${SKIP_APPENDIX:-0}" != "1" ]]; then
  echo "== [4/4] 附录 risk–coverage 图"
  "${PY}" "${APPENDIX}" \
    --search-root "${SEARCH_ROOT}" \
    --analysis-dir "${ANALYSIS_DIR}" \
    --out-dir "${SEARCH_ROOT}/appendix_extra_pairs" \
    --proxy "${PROXY}" \
    --n-pairs 5
else
  echo "== [4/4] SKIP_APPENDIX=1，跳过附录"
fi

echo "完成。主图: ${SEARCH_ROOT}/killer_final.png"
echo "分析目录: ${ANALYSIS_DIR}"
